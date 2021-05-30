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
    def __init__(self, env_fn, actor_critic, sampler, ac_kwargs, seed, 
         steps_per_epoch, replay_size, gamma, 
         polyak, pi_lr, q_lr, batch_size, noise_dim, n_particles, start_steps, 
         update_after, update_every, act_noise, num_test_episodes, 
         max_ep_len):

        self.env= MultiGoalEnv()
        
        self.gamma = gamma
        self.num_test_episodes = num_test_episodes
        self.n_particles = n_particles
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len

        
        
        self.env, self.test_env = self.env, self.env
        self.obs_dim = self.env.observation_space.shape
        
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        
        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        #self.ac_targ = deepcopy(self.ac)
        
        self.sampler = sampler(noise_dim = noise_dim, batch_size = self.batch_size, n_particles = self.n_particles,
                    observation_space= self.env.observation_space, action_space = self.env.action_space)
        # # Freeze target networks with respect to optimizers (only update via polyak averaging)
        # for p in self.ac_targ.parameters():
        #     p.requires_grad = False
        
        # Set up optimizers for policy and q-function
        self.pi_optimizer =optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = optim.Adam(self.ac.q.parameters(), lr=q_lr)
        self.sampler_optimizer = optim.Adam(self.sampler.parameters(), lr=pi_lr)
        
        self.polyak = polyak
        
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
            q_pi_targ = T.logsumexp(self.ac.q(o2, aplus,n_sample=self.n_particles), dim=1)
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
    
    
    
    def update_svgd_ss(self, data):
        
        o = data['obs']
        actions = self.sampler(o,n_particles=self.n_particles)
        assert actions.shape == (self.batch_size,self.n_particles, self.action_dim)
        
        fixed_actions = self.sampler.act(o,n_particles=self.n_particles)
        fixed_actions = T.from_numpy(fixed_actions)
        fixed_actions.requires_grad = True
        svgd_target_values = self.ac.q(o, fixed_actions,n_sample = self.n_particles)
    
        # Target log-density. Q_soft in Equation 13:
        
        log_p = svgd_target_values
    
    
        grad_log_p = T.autograd.grad(log_p.sum(), fixed_actions)[0]
        grad_log_p = grad_log_p.view(self.batch_size,self.n_particles, self.action_dim).unsqueeze(2)
        grad_log_p = grad_log_p.detach()
        assert grad_log_p.shape == (self.batch_size, self.n_particles, 1, self.action_dim)
    
        kappa, gradient = self.rbf_kernel(input_1=fixed_actions, input_2=actions)
    
        # Kernel function in Equation 13:
        # kappa = kappa.unsqueeze(dim=3)
        assert kappa.shape == (self.batch_size, self.n_particles, self.n_particles, 1)
    
       
        anneal = 1.
        action_gradients = (1/self.n_particles)*T.sum(anneal*kappa * grad_log_p + gradient, dim=1)
        assert action_gradients.shape == (self.batch_size, self.n_particles, self.action_dim)
    
        # Propagate the gradient through the policy network (Equation 14).
        self.sampler_optimizer.zero_grad()
        T.autograd.backward(-actions,grad_tensors=action_gradients)
        self.sampler_optimizer.step()
    
    def learn(self, data):
    
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
    
        # update stein sampler 
        self.update_svgd_ss(data)
        #This PLACE
      
    
    def get_action(self, o, noise_scale):
        a = self.ac.act(T.as_tensor(o, dtype=T.float32))
        return np.clip(a, -self.action_bound, self.action_bound)
    
     
    def get_sample(self, o,n_sample=1):
        a = self.sampler.act(T.as_tensor(o, dtype=T.float32),n_particles=n_sample)
        # a += act_noise * np.random.randn(act_dim)
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
            while not done and step < self.max_ep_len :
                
                actions = self.get_sample(observation,1)
               
                observation, reward, done, _ = env.step(actions)
                path['infos']['pos'].append(observation)
                step +=1
                paths.append(path)
        print("saving figure..., epoch=",epoch)
            #         with open('./actions_'+str(epoch)+'.txt', 'w') as filehandle:
            #             for listitem in actions_plot:
            #                 filehandle.write('%s\n' % listitem) 
        env.render_rollouts(paths,fout="test_%d.png" % epoch)
    
    


# class Agent():
#     def __init__(self, beta=0.0003, state_dim=[8], action_dim=2, n_particles=16,
#             env=None, gamma=0.99, max_size=int(1e6), tau=0.005, max_action=1000,
#             batch_size=100, reward_scale=1):
        
#         self.gamma = gamma
#         self.tau = tau
#         self.memory = ReplayBuffer(max_size, state_dim, action_dim)
#         self.batch_size = batch_size
#         self.action_dim = action_dim
#         self.n_particles = n_particles
        
#         self.update_ratio = 0.5


#         # Q Network
#         self.Q_Network = ActionValueNetwork(lr=1e-3, state_dim=state_dim, action_dim=action_dim,
#                     n_particles=n_particles, name='ActionValueNetwork')
#         self.Q_Network.double()
#         # q Arbitrary Network
#         self.SVGD_Network = SamplerNetwork(lr=1e-3, state_dim=state_dim, action_dim=action_dim,
#                     n_particles=n_particles, max_action=max_action)
#         self.SVGD_Network.double()
#         self.reward_scale = reward_scale

#     def remember(self, state, action, reward, new_state, done):
#         self.memory.store_transition(state, action, reward, new_state, done)

#     def save_models(self):
#         print('.... saving models ....')
#         self.Q_Network.save_checkpoint()
#         self.SVGD_Network.save_checkpoint()
        

#     def load_models(self):
#         print('.... loading models ....')
#         self.Q_Network.save_checkpoint()
#         self.SVGD_Network.save_checkpoint()
    
    
    
#     def rbf_kernel(self, input_1, input_2,  h_min=1e-3):
        
#         k_fix, out_dim1 = input_1.size()[-2:]
#         k_upd, out_dim2 = input_2.size()[-2:]
#         assert out_dim1 == out_dim2
    
#         leading_shape = input_1.size()[:-2]
#         # Compute the pairwise distances of left and right particles.
#         diff = input_1.unsqueeze(-2) - input_2.unsqueeze(-3)
#         # N * k_fix * 1 * out_dim / N * 1 * k_upd * out_dim/ N * k_fix * k_upd * out_dim
#         dist_sq = diff.pow(2).sum(-1)
#         # N * k_fix * k_upd
#         dist_sq = dist_sq.unsqueeze(-1)
#         # N * k_fix * k_upd * 1
    
#         # Get median.
#         median_sq = T.median(dist_sq, dim=1)[0]
#         median_sq = median_sq.unsqueeze(1)
#         # N * 1 * k_upd * 1
    
#         h = median_sq / np.log(k_fix + 1.) + .001
#         # N * 1 * k_upd * 1
    
#         kappa = T.exp(-dist_sq / h)
#         # N * k_fix * k_upd * 1
    
#         # Construct the gradient
#         kappa_grad = -2. * diff / h * kappa
#         return kappa, kappa_grad
    
        
        
#     def get_Q_value(self, state, action, particles=False, training=False): 
#         # (bs, sd), (bs, np, ad)
        
        
#         if training and particles:
#             # Training with particles
#             Q_soft = self.Q_Network.forward(state.unsqueeze(1).double(), action.double())
        
        
#         elif training and not particles:
#             # Training without particles
#             Q_soft = self.Q_Network.forward(state.double(), action.double())
        
#         elif not training and particles:
#             # Evaluating with particles
#             Q_soft = self.Q_Network.forward(state.double(), action.double())
        
#         return Q_soft
        
        
        
    
#     def choose_action_uniform(self, particles=False, reparameterize=True):
#         if particles:
#             low = T.full((self.batch_size, self.n_particles, self.action_dim), -1.)
#             high = T.full((self.batch_size, self.n_particles, self.action_dim), 1.)
#         else:
#             low = T.full((self.batch_size, self.action_dim), -1.)
#             high = T.full((self.batch_size, self.action_dim), 1.)
#         dist = Uniform(low, high)
#         noise = dist.sample()
#         return noise, dist
    
    
#     def get_action_svgd(self, state, training=False, particles=False):
#         # Sample noise from  normal  distribution
        
#         if training and particles:
#             low = T.full((self.batch_size, self.n_particles, self.action_dim), -1.)
#             high = T.full((self.batch_size, self.n_particles, self.action_dim), 1.)
#             dist = Uniform(low, high)
#             noise = dist.sample()
#             actions = self.SVGD_Network.forward(state.double().unsqueeze(1), noise.double())
#         elif training and not particles:
#             low = T.full((self.batch_size, self.action_dim), -1.)
#             high = T.full((self.batch_size, self.action_dim), 1.)
#             dist = Uniform(low, high)
#             noise = dist.sample()
#             #print(noise)
#             actions = self.SVGD_Network.forward(state.double(), noise.double())
#         elif not training and not particles:
#             low = T.full((1, 1, self.action_dim), -1.)
#             high = T.full((1, 1, self.action_dim), 1.)
#             dist = Uniform(low, high)
#             noise = dist.sample()
#             self.SVGD_Network.eval()
#             actions = self.SVGD_Network.forward(state.double().unsqueeze(0), noise.double())
#         elif not training and particles:
#             low = T.full((1, self.n_particles, self.action_dim), -1.)
#             high = T.full((1, self.n_particles, self.action_dim), 1.)
#             dist = Uniform(low, high)
#             noise = dist.sample()
#             self.SVGD_Network.eval()
#             actions = self.SVGD_Network.forward(state.double().unsqueeze(0), noise.double())
#         return actions

        
                
        
        
# ############################ LEARN ###############################    
#     def learn(self, steps):
#         if self.memory.mem_cntr < self.batch_size:
#             return
        
#         # Sample a minibatch from the replay memory
#         state, action, reward, new_state, done = \
#                 self.memory.sample_buffer(self.batch_size)
        
#         # -------- Update the soft Q-function Parameters -------- #
        
#         # Converting the sampled experience to Tensors
#         state = T.tensor(state, dtype=T.float).to(self.SVGD_Network.device)
#         action = T.tensor(action, dtype=T.float).to(self.SVGD_Network.device)
#         reward = T.tensor(reward, dtype=T.float).to(self.SVGD_Network.device)
#         state_ = T.tensor(new_state, dtype=T.float).to(self.SVGD_Network.device)
#         done = T.tensor(done.astype(float)).to(self.SVGD_Network.device)
        
#         # for i in range(10):
#         #     print('state : ', state[i].numpy().squeeze(), ' action : ', action[i].numpy().squeeze(), ' reward : ', reward[i].numpy().squeeze(), ' state_ : ', state_[i].numpy().squeeze())
                
#         # Sample actions for next states (state_) (bs, np, ad)
#         #action_, dist = self.choose_action_uniform(state_)
#         action_ = self.get_action_svgd(state, training=True, particles=True)
                
#         # Calculate the Q-value using the Q-network for next states (state_) (bs, np, 1)
#         Q_soft_ = self.get_Q_value(state_, action_, training=True, particles=True)
#         print(Q_soft_)
        
        
#         # Equation 10 
#         V_soft_ = T.logsumexp(Q_soft_, dim=1)
#         V_soft_ += self.action_dim * T.log(T.tensor([2.]))
        
        
#         # Evaluate Q hat in Equation 11
#         with T.no_grad():
#             Q_soft_hat = self.reward_scale * reward.unsqueeze(-1) + self.gamma * (1 - done.unsqueeze(-1)) * V_soft_ # (bs, 1)
        
#         # Calculate the Q-value using the Q-network for current states (state) 
#         Q_soft = self.get_Q_value(state, action, training=True, particles=False) # (bs, np)


#         # Equation 11 
#         l2 = 0.0
#         for p in self.Q_Network.parameters():
#             l2 = l2 + p.norm(2)
#         J_Q = 0.5 * T.mean((Q_soft_hat - Q_soft) ** 2, dim=0) + l2
        
#         print(J_Q)
        
#         # Update Q Network 
#         Q_network_loss = J_Q
#         self.Q_Network.optimizer.zero_grad()
#         Q_network_loss.backward()
#         self.Q_Network.optimizer.step()
        
#         # -------- Update The Policy -------- #
        
#         if steps%30 == 0:
#             # Compute aciton    
#             action_svgd = self.get_action_svgd(state, training=True, particles=True) # (bs, np, ad)
#             #print('action_svgd', action_svgd)
#             #print(state)
#             svgd_Q_soft = self.get_Q_value(state, action_svgd, training=True, particles=True) # (bs, np, 1)
#             # print('action_svgd : ', action_svgd.squeeze(-1))
#             # print('svgd_Q_soft : ', svgd_Q_soft)
    
#             squash_correction = T.sum(T.log(1 - action_svgd**2 + 1e-6), dim=-1)
#             svgd_Q_soft = T.add(svgd_Q_soft, squash_correction.unsqueeze(-1))
            
            
#             # Get the Gradients of the energy with respect to x and y
#             grad_score = T.autograd.grad(svgd_Q_soft.sum(), action_svgd)[0].squeeze(-1)
           
#             # Compute the similarity using the RBF kernel 
#             # kappa grad_kappa= T.empty(1)
#             # for i in range(self.batch_size): 
#             kappa, grad_kappa = self.rbf_kernel(input_1=action_svgd, input_2=action_svgd) # (bs, np, ad)
                
        
            
#             l2 = 0.0
#             for p in self.SVGD_Network.parameters():
#                 l2 = l2 + p.norm(2)
#             svgd = T.sum(kappa * grad_score.unsqueeze(2) + grad_kappa, dim=1) / self.n_particles - l2 # (bs, np * ad)
#             #print('svgd : ' ,svgd)
#             self.SVGD_Network.optimizer.zero_grad()
#             T.autograd.backward(-action_svgd, grad_tensors=svgd)
#             self.SVGD_Network.optimizer.step()  
        
        
      
        
        
   
        
        
        