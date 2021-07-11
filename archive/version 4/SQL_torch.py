import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
#from networks import ActionValueNetwork, SamplerNetwork
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from networks import MLPActorCritic, SamplingNetwork
import torch.optim as optim
import time
from copy import deepcopy


from multigoal import MultiGoalEnv






class Agent():
    def __init__(self, env_fn, actor_critic, sampler, hidden_dim, replay_size, gamma, pi_lr, q_lr, batch_size, n_particles):

# svgd(lambda : gym.make(env) , actor_critic=MLPActorCritic, ac_kwargs=dict(hidden_sizes=[hid]*l), 
#       gamma=gamma, seed=seed, epochs=epochs)

        self.env= MultiGoalEnv()
        # T.manual_seed(seed)
        # np.random.seed(seed)
        
        self.gamma = gamma
        self.n_particles = n_particles
        self.batch_size = batch_size

        
        
        #env, test_env = env_fn(), env_fn()
        self.env, self.test_env = self.env, self.env
        self.obs_dim = self.env.observation_space.shape
        
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        #print(f'env.action_space.shape ={env.action_space.shape }')
        
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        
        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, hidden_dim)
        self.ac_targ = deepcopy(self.ac)
        
        self.sampler = sampler(batch_size = self.batch_size, n_particles = self.n_particles,
                    observation_space= self.env.observation_space, action_space = self.env.action_space)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        
        # Set up optimizers for policy and q-function
        self.pi_optimizer =optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = optim.Adam(self.ac.q.parameters(), lr=q_lr)
        self.sampler_optimizer = optim.Adam(self.sampler.parameters(), lr=pi_lr)
        
        
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.action_dim, size=replay_size)

    def rbf_kernel(self, input_1, input_2,  h_min=1e-3):
        k_fix, out_dim1 = input_1.size()[-2:]
        k_upd, out_dim2 = input_2.size()[-2:]
        assert out_dim1 == out_dim2
        
        # Compute the pairwise distances of left and right particles.
        diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
        # N * k_fix * 1 * out_dim / N * 1 * k_upd * out_dim/ N * k_fix * k_upd * out_dim
        dist_sq = diff.pow(2).sum(-1)
        # N * k_fix * k_upd
        dist_sq = dist_sq.unsqueeze(-1)
        # N * k_fix * k_upd * 1
        
        # Get median.
        median_sq = T.median(dist_sq, dim=1)[0]
        median_sq = median_sq.unsqueeze(1)
        # N * 1 * k_upd * 1
        
        h = median_sq / np.log(k_fix + 1.) + .001
        # N * 1 * k_upd * 1
        
        kappa = T.exp(-dist_sq / h)
        # N * k_fix * k_upd * 1
        
        # Construct the gradient
        kappa_grad = -2. * diff / h * kappa
        return kappa, kappa_grad
    
    
    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
    
        aplus = T.from_numpy(self.sampler.act(o2,n_particles=self.n_particles))
        #print("aplus=",aplus.shape)
        
        q = self.ac.q(o,a)
        
    
        # Bellman backup for Q function
        with T.no_grad():
            #q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            q_pi_targ = T.logsumexp(self.ac_targ.q(o2, aplus,n_sample=self.n_particles), dim=1)
            #q_pi_targ += action * T.log(T.tensor([2.]))
            backup = r + self.gamma * (1 - d) * q_pi_targ
    
        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()
    
        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())
    
        # print(f'loss_q: {loss_q}')
        # print(f'q_pi_targ: {q_pi_targ}')
        # print(f'q1: {q}')
    
        return loss_q, loss_info
    
    # Set up function for computing svgd pi loss
    def compute_loss_pi_svgd(self, data):
        o = data['obs']
        ss_targ = T.from_numpy(self.sampler.act(o))
        ss_targ.requires_grad = False
        a = self.ac.pi(o)
        # a = a.unsqueeze(dim=1)
          # MSE loss against Sampling policy
        l2 = 0.0
        # for p in ac.q.parameters():
        #     l2 = l2 + p.norm(2)
        loss = ((a - ss_targ)**2).mean() + 0.1*l2
    
        # Useful info for logging
        loss_info = dict(AVals=a.detach().numpy())
    
     
        return loss, loss_info
    
    # Set up function for computing DDPG pi loss
    
    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()
    
    
    
    
    # Set up model saving
    #logger.setup_pyT_saver(ac)
    
    
    def update_svgd_ss(self, data,train=True):
        o = data['obs']
        actions = self.sampler(o,n_particles=self.n_particles)
        assert actions.shape == (self.batch_size,self.n_particles, self.action_dim)
        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        # n_fixed_actions = n_updated_actions = n_particles//2
        # fixed_actions, updated_actions = T.split(actions, [n_fixed_actions, n_updated_actions], dim=1)
        # fixed_actions.detach()
        # fixed_actions.requires_grad = True
        fixed_actions = self.sampler.act(o,n_particles=self.n_particles)
        fixed_actions = T.from_numpy(fixed_actions)
        fixed_actions.requires_grad = True
        svgd_target_values = self.ac.q(o, fixed_actions,n_sample = self.n_particles)
    
          # Target log-density. Q_soft in Equation 13:
        # squash_correction = T.sum(
        #     T.log(1 - fixed_actions**2 + 1e-6), dim=-1)
        # log_p = svgd_target_values + squash_correction
        log_p = svgd_target_values
    
    
        grad_log_p = T.autograd.grad(log_p.sum(), fixed_actions)[0]
        grad_log_p = grad_log_p.view(self.batch_size,self.n_particles, self.action_dim).unsqueeze(2)
        grad_log_p = grad_log_p.detach()
        assert grad_log_p.shape == (self.batch_size, self.n_particles, 1, self.action_dim)
    
        kappa, gradient = self.rbf_kernel(input_1=fixed_actions, input_2=actions)
    
          # Kernel function in Equation 13:
        # kappa = kappa.unsqueeze(dim=3)
        assert kappa.shape == (self.batch_size, self.n_particles, self.n_particles, 1)
    
        # Stein Variational Gradient in Equation 13:
        # T_C = total_steps/steps_per_epoch
        # anneal = ((t%(T_C))/T_C)**2
        anneal = 1.
        action_gradients = (1/self.n_particles)*T.sum(anneal*kappa * grad_log_p + gradient, dim=1)
        assert action_gradients.shape == (self.batch_size, self.n_particles, self.action_dim)
    
        # Propagate the gradient through the policy network (Equation 14).
        if train:
            self.sampler_optimizer.zero_grad()
            T.autograd.backward(-actions,grad_tensors=action_gradients)
            #T.nn.utils.clip_grad_norm_(g_net.parameters(),2)
            self.sampler_optimizer.step()
    
    def learn(self, data):
    
        # update stein sampler 
        self.update_svgd_ss(data)
        #print("done update_svgd_ss")
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
    
      
        
    def get_action(self, o, noise_scale):
        a = self.ac.act(T.as_tensor(o, dtype=T.float32))
        return np.clip(a, -self.action_bound, self.action_bound)
    
     
    def get_sample(self, o,n_sample=1):
        a = self.sampler.act(T.as_tensor(o, dtype=T.float32),n_particles=n_sample)
        return np.clip(a, -self.action_bound, self.action_bound) 
    
    
    def plot_paths(self, epoch):
        paths = []
        actions_plot=[]
        env = MultiGoalEnv()
    
        for episode in range(50):
            observation = env.reset()
            done = False
            step = 0
            path = {'infos':{'pos':[]}}
            particles = None
            while not done and step < 30 :
                #a = get_action(observation,0)
                #print("a.shape=",a.shape)
                actions = self.get_sample(observation,1)
                # a = b[0]
                #print("before b.shape=",b.shape, "a.shape=",a.shape)
                #             print("a=",a)
                # if (observation[0]<=2.6) & (observation[0]>=2.4)& (observation[1]>=2.4)&(observation[1]<=2.6):
                #    actions_plot.append(a)
                # print(f'actions.shape={actions.shape}')
                
                observation, reward, done, _ = env.step(actions)
                path['infos']['pos'].append(observation)
                # print(f'observation.shape={observation.shape}')
                step +=1
                paths.append(path)
        print("saving figure..., epoch=",epoch)
            #         with open('./actions_'+str(epoch)+'.txt', 'w') as filehandle:
            #             for listitem in actions_plot:
            #                 filehandle.write('%s\n' % listitem) 
        env.render_rollouts(paths,fout="test_%d.png" % epoch)
    