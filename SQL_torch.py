import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActionValueNetwork, SamplerNetwork
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal


class Agent():
    def __init__(self, beta=0.0003, state_dim=[8], action_dim=2, n_particles=16,
            env=None, gamma=0.99, max_size=int(1e6), tau=0.005, max_action=1000,
            batch_size=100, reward_scale=2):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, state_dim, action_dim)
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.n_particles = n_particles
        
        self.update_ratio = 0.5


        # Q Network
        self.Q_Network = ActionValueNetwork(lr=3e-4, state_dim=state_dim, action_dim=action_dim,
                    n_particles=n_particles, name='ActionValueNetwork')
        self.Q_Network.double()
        # q Arbitrary Network
        self.SVGD_Network = SamplerNetwork(lr=3e-4, state_dim=state_dim, action_dim=action_dim,
                    n_particles=n_particles, max_action=max_action)
        self.SVGD_Network.double()
        self.reward_scale = reward_scale

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
    
    
    
    def rbf_kernel(self, input_1, input_2,  h_min=1e-3):
        
        k_fix, out_dim1 = input_1.size()[-2:]
        k_upd, out_dim2 = input_2.size()[-2:]
        assert out_dim1 == out_dim2
    
        leading_shape = input_1.size()[:-2]
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
    
        
        
    def get_Q_value(self, state, action, particles=False, training=False): 
        # (bs, sd), (bs, np, ad)
        
        
        if training and particles:
            # Training with particles
            Q_soft = self.Q_Network.forward(state.unsqueeze(1).double(), action.double())
        
        
        elif training and not particles:
            # Training without particles
            Q_soft = self.Q_Network.forward(state.double(), action.double())
        
        elif not training and particles:
            # Evaluating with particles
            Q_soft = self.Q_Network.forward(state.double(), action.double())
        
        return Q_soft
        
        
        
    
    def choose_action_uniform(self, reparameterize=True):
        low = T.full((self.batch_size, self.n_particles, self.action_dim), -1.)
        high = T.full((self.batch_size, self.n_particles, self.action_dim), 1.)
        # low = T.full((self.batch_size, self.action_dim), -1.)
        # high = T.full((self.batch_size, self.action_dim), 1.)
        dist = Uniform(low, high)
        noise = dist.sample()
        return noise, dist
    
    
    def get_action_svgd(self, state, training=False, particles=False):
        # Sample noise from  normal  distribution
        
        if training and particles:
            low = T.full((self.batch_size, self.n_particles, self.action_dim), -1.)
            high = T.full((self.batch_size, self.n_particles, self.action_dim), 1.)
            dist = Uniform(low, high)
            noise = dist.sample()
            #print(noise)
            actions = self.SVGD_Network.forward(state.double().unsqueeze(1), noise.double())
        elif not training and not particles:
            low = T.full((1, 1, self.action_dim), -1.)
            high = T.full((1, 1, self.action_dim), 1.)
            dist = Uniform(low, high)
            noise = dist.sample()
            self.SVGD_Network.eval()
            actions = self.SVGD_Network.forward(state.double().unsqueeze(0), noise.double())
        elif not training and particles:
            low = T.full((1, self.n_particles, self.action_dim), -1.)
            high = T.full((1, self.n_particles, self.action_dim), 1.)
            dist = Uniform(low, high)
            noise = dist.sample()
            self.SVGD_Network.eval()
            actions = self.SVGD_Network.forward(state.double().unsqueeze(0), noise.double())
        return actions

        
                
        
        
############################ LEARN ###############################    
    def learn(self, steps):
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
        Q_soft_ = self.get_Q_value(state_, action_, training=True, particles=True)
        #print(Q_soft_)
        
        
        # Equation 10 
        V_soft_ = T.logsumexp(Q_soft_, dim=1)
        V_soft_ += self.action_dim * T.log(T.tensor([2.]))
        
        
        # V_soft_ = self.alpha * T.log(
        #         T.div(
        #             T.sum(
        #                 T.div(
        #                     T.exp(
        #                         T.multiply(Q_soft_, 1/self.alpha)  # (bs, np, 1)
        #                     ),
        #                     dist.cdf(action_).mean(-1).unsqueeze(-1) # (bs, np, 1)
        #                 ), 
        #                 dim=1 # sum over np => (bs, 1)
        #             ),
        #             self.batch_size # (bs, 1)
        #         )
        #     ) #(bs, 1)
        
        # Evaluate Q hat in Equation 11
        with T.no_grad():
            Q_soft_hat = self.reward_scale * reward.unsqueeze(-1) + self.gamma * (1 - done.unsqueeze(-1)) * V_soft_ # (bs, 1)
            #Q_soft_hat = self.reward_scale * reward.unsqueeze(-1) + self.gamma * (1 - done.unsqueeze(-1)) * Q_soft_ # (bs, 1)
        #print(Q_soft_hat)
        # Calculate the Q-value using the Q-network for current states (state) 
        Q_soft = self.get_Q_value(state, action, training=True, particles=False) # (bs, np)
        #print('Q', Q_soft)


        # Equation 11 
        J_Q = 0.5 * T.mean((Q_soft_hat - Q_soft) ** 2, dim=0)
        
        #print(J_Q)
        
        # Update Q Network 
        Q_network_loss = J_Q
        self.Q_Network.optimizer.zero_grad()
        Q_network_loss.backward()
        self.Q_Network.optimizer.step()
        
        # -------- Update The Policy -------- #
        
        if steps%1 == 0:
            # Compute aciton    
            action_svgd = self.get_action_svgd(state, training=True, particles=True) # (bs, np, ad)
            #print('action_svgd', action_svgd)
            #print(state)
            svgd_Q_soft = self.get_Q_value(state, action_svgd, training=True, particles=True) # (bs, np, 1)
            # print('action_svgd : ', action_svgd.squeeze(-1))
            # print('svgd_Q_soft : ', svgd_Q_soft)
    
            squash_correction = T.sum(T.log(1 - action_svgd**2 + 1e-6), dim=-1)
            svgd_Q_soft = T.add(svgd_Q_soft, squash_correction.unsqueeze(-1))
    
            #print(action_svgd_2d)
            #print(svgd_Q_soft)
            # Get the Gradients of the energy with respect to x and y
            grad_score = T.autograd.grad(svgd_Q_soft.sum(), action_svgd)[0].squeeze(-1)
            #print(grad_score)# (bs, np * ad)
           
            # Compute the similarity using the RBF kernel 
            # kappa grad_kappa= T.empty(1)
            # for i in range(self.batch_size): 
            kappa, grad_kappa = self.rbf_kernel(input_1=action_svgd, input_2=action_svgd) # (bs, np, ad)
                
        
            # print(a.mean())
            # print(grad_kappa.mean())
            svgd = T.sum(kappa * grad_score.unsqueeze(2)  * 1000 + grad_kappa, dim=1) / self.n_particles # (bs, np * ad)
            #print('svgd : ' ,svgd)
            self.SVGD_Network.optimizer.zero_grad()
            T.autograd.backward(-action_svgd, grad_tensors=svgd)
            self.SVGD_Network.optimizer.step()  
        
        
        
        # # Target log-density. Q_soft in Equation 13:
        
        # squash_correction = T.sum(T.log(1 - fixed_actions ** 2 + 1e-6), axis=-1)
        
        # log_p = svgd_target_values + squash_correction
        
        
        
        