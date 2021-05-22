import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActionValueNetwork, SamplerNetwork
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal


class Agent():
    def __init__(self, alpha=0.5, beta=0.0003, state_dim=[8], action_dim=2, n_particles=16,
            env=None, gamma=0.99, max_size=1000000, tau=0.005, max_action=1000,
            batch_size=100, reward_scale=2):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, state_dim, action_dim)
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.alpha = alpha
        self.n_particles = n_particles
        
        self.update_ratio = 0.5


        # Q Network
        self.Q_Network = ActionValueNetwork(alpha, state_dim=state_dim, action_dim=action_dim,
                    n_particles=n_particles, name='ActionValueNetwork')
        # q Arbitrary Network
        self.SVGD_Network = SamplerNetwork(alpha, state_dim=state_dim, action_dim=action_dim,
                    n_particles=n_particles, max_action=max_action)
        
        self.scale = reward_scale

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('.... saving models ....')
        self.Q_Network.save_checkpoint()
        self.SVGD_Network.save_checkpoint()
        

    def load_models(self):
        print('.... loading models ....')
        self.Q_Network.save_checkpoint()
        self.SVGD_Network.save_checkpoint()
    
    def rbf_kernel(self, X, Y,  h_min=1e-3):

        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())
    
        dnorm2 = XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0) - 2 * XY
    
        # Apply the median heuristic (PyTorch does not give true median)
        np_dnorm2 = dnorm2.detach().cpu().numpy()
        h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
        sigma = np.sqrt(h).item()
        #print('sigma ------------- :', sigma)
        #sigma = 0.1
    
        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()
        
        grad_K = -T.autograd.grad(K_XY.mean(), X)[0]
        
        return K_XY, grad_K
        
        
    def get_Q_value(self, state, action, particles=False, training=False): 
        # (bs, sd), (bs, np, ad) => (bs, sd + np * ad)
        nn_input = []
        state = state.numpy()
        action = action.numpy()
        
        if training and particles:
            # Training with particles 
            for i in range(self.batch_size):
                s_tmp = state[i]
                a_tmp = action[i, :].T
                s_tmp = np.concatenate((s_tmp, a_tmp), axis=None)
                nn_input.append(s_tmp) 
        elif training and not particles:
            # Training without particles
            for i in range(self.batch_size):
                s_tmp = state[i]
                a_tmp = action[i, :]
                for j in range(self.n_particles):
                    s_tmp = np.concatenate((s_tmp, a_tmp), axis=None)
                nn_input.append(s_tmp)
        elif not training and particles:
            # Evaluating with particles
            s_tmp = state
            a_tmp = action
            s_tmp = np.concatenate((s_tmp, a_tmp), axis=None)
            nn_input.append(s_tmp) 
                
        nn_input = T.tensor(np.array(nn_input, dtype=np.float32), requires_grad=True)
        #print(nn_input)
        Q_soft = self.Q_Network.forward(nn_input)
        #print(Q_soft)

        return nn_input, Q_soft
        
    
    def choose_action_uniform(self, reparameterize=True):
        low = T.full((self.batch_size, self.n_particles, self.action_dim), -1.)
        high = T.full((self.batch_size, self.n_particles, self.action_dim), 1.)
        dist = Uniform(low, high)
        noise = dist.sample()
        return noise, dist
    
    
    def choose_action_svgd(self, state, training=False):
        # Sample noise from  normal  distribution
        if not training:
            mu = T.from_numpy(np.zeros((1, self.n_particles * self.action_dim)))
            sigma = T.from_numpy(np.ones((1, self.n_particles * self.action_dim)))
            noise = Normal(mu, sigma).sample()
        else:
            mu = T.from_numpy(np.zeros((self.batch_size, self.n_particles * self.action_dim)))
            sigma = T.from_numpy(np.ones((self.batch_size, self.n_particles * self.action_dim)))
            noise = Normal(mu, sigma).sample()
        
        # if self.action_dim == 1:
        #     noise = noise.unsqueeze(-1)
        
        
        
        nn_input = []
        for i in range(noise.shape[0]):
            nn_input.append(np.concatenate((state[i], noise[i]), axis=None))
        nn_input = T.from_numpy(np.array(nn_input, dtype=np.float32))
        
        if not training:
            self.SVGD_Network.eval()
        y = self.SVGD_Network.forward(nn_input)
        
        if not training:
            particles = []
            s = 0
            f = self.action_dim
            for _ in range(self.n_particles):
                particles.append(list(y[0, s:f].detach().numpy()))
                s += self.action_dim
                f += self.action_dim
            return T.from_numpy(np.array(particles, dtype=np.float32))
            
        else:
            instances = []
            
            for i in range(self.batch_size):
                particles = []
                s = 0
                f = self.action_dim
                for j in range(self.n_particles):
                    particles.append(list(y[i,s:f]))
                    s += self.action_dim
                    f += self.action_dim
                instances.append(particles)
            return T.from_numpy(np.array(instances, dtype=np.float32))
                
        
        
############################ LEARN ###############################    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        # Sample a minibatch from the replay memory
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)
        
        # -------- Update the soft Q-function Parameters -------- #
        
        # Converting the sampled experience to Tensors
        state = T.tensor(state, dtype=T.float).to(self.SVGD_Network.device)
        action = T.tensor(action, dtype=T.float).to(self.SVGD_Network.device)
        reward = T.tensor(reward, dtype=T.float).to(self.SVGD_Network.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.SVGD_Network.device)
        done = T.tensor(done.astype(float)).to(self.SVGD_Network.device)
        
        # Sample actions for next states (state_) (bs, np, ad)
        action_, dist = self.choose_action_uniform(state_)
                
        # Calculate the Q-value using the Q-network for next states (state_) (bs, np, 1)
        _, Q_soft_ = self.get_Q_value(state_, action_, training=True, particles=True)
        # print(Q_soft_)
        # print(dist.cdf(action_))
        
        
        a = dist.cdf(action_).mean(-1)
        
        
        # Equation 10 
        V_soft_ = self.alpha * T.log(
                T.div(
                    T.sum(
                        T.div(
                            T.exp(
                                T.multiply(Q_soft_.unsqueeze(-1), 1/self.alpha)  # (bs, np, 1)
                            ),
                            dist.cdf(action_).mean(-1).unsqueeze(-1) # (bs, np, 1)
                        ), 
                        dim=1 # sum over np => (bs, 1)
                    ),
                    self.batch_size # (bs, 1)
                )
            ) #(bs, 1)
        # print(V_soft_)
        # Evaluate Q hat in Equation 11
        Q_soft_hat = reward.unsqueeze(-1) + self.gamma * (1 - done.unsqueeze(-1)) * V_soft_ # (bs, 1)
        # print(Q_soft_hat)
        
        # Calculate the Q-value using the Q-network for current states (state) 
        _, Q_soft = self.get_Q_value(state, action, training=True) # (bs, np)
        
        # Not sure about this
        #Q_soft = T.mean(Q_soft, dim=1).unsqueeze(-1) # (bs, 1)
        #Q_soft = Q_soft[:, 0].unsqueeze(-1) # (bs, 1)
        Q_soft = Q_soft.max().unsqueeze(-1)
        # print(Q_soft)

        # Equation 11 
        J_Q = 0.5 * T.mean((Q_soft_hat - Q_soft) ** 2, dim=0)
        
        # print(J_Q)
        
        # Update Q Network 
        Q_network_loss = J_Q
        self.Q_Network.optimizer.zero_grad()
        Q_network_loss.backward(retain_graph=True)
        self.Q_Network.optimizer.step()
        
        # -------- Update The Policy -------- #
        
        # Compute aciton    
        action_svgd = self.choose_action_svgd(state, training=True) # (bs, np, ad)
        #print(state)
        action_svgd_2d, svgd_Q_soft = self.get_Q_value(state, action_svgd, training=True, particles=True) # (bs, np, 1)
        # print('action_svgd : ', action_svgd.squeeze(-1))
        # print('svgd_Q_soft : ', svgd_Q_soft)

        svgd_Q_soft = svgd_Q_soft.mean(-1)

        #print(action_svgd_2d)
        #print(svgd_Q_soft)
        # Get the Gradients of the energy with respect to x and y
        grad_score = T.autograd.grad(svgd_Q_soft.sum(), action_svgd_2d)[0].squeeze(-1)
        #print(grad_score)# (bs, np * ad)
       
        # Compute the similarity using the RBF kernel 
        kappa, grad_kappa = self.rbf_kernel(action_svgd_2d, action_svgd_2d) # Still not fixed # (bs, np * ad)
    
        # print(a.mean())
        # print(grad_kappa.mean())
        svgd = (T.matmul(kappa.squeeze(-1), grad_score) + grad_kappa) / action_svgd_2d.size(0) # (bs, np * ad)
        #print('svgd : ' ,svgd)
        self.SVGD_Network.optimizer.zero_grad()
        T.autograd.backward(-action_svgd_2d, grad_tensors=svgd)
        self.SVGD_Network.optimizer.step()  
        
        
        
        # # Target log-density. Q_soft in Equation 13:
        
        # squash_correction = T.sum(T.log(1 - fixed_actions ** 2 + 1e-6), axis=-1)
        
        # log_p = svgd_target_values + squash_correction
        
        
        
        