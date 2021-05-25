import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act(), nn.Dropout(0.2)]
    return nn.Sequential(*layers)


class MLPQFunction(nn.Module):

    def __init__(self, observation_space, action_space, hidden_dim, activation=nn.ReLU):
        super().__init__()
        
        # self.state_layer = nn.Linear(obs_dim,hidden_sizes[0])
        # self.action_layer = nn.Linear(act_dim,hidden_sizes[0])
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.q = mlp([self.obs_dim + self.act_dim] + hidden_dim + [1], activation)

    def forward(self, obs, act,n_sample = 1):
        if n_sample > 1:
            obs = obs.unsqueeze(1).repeat(1,n_sample,1)
            assert obs.dim() == 3
        q = self.q(T.cat([act,obs], dim=-1))
        return  T.squeeze(q, -1).squeeze(-1)   # Critical to ensure q has right shape.



class SamplingNetwork(nn.Module):
    def __init__(self,noise_dim,n_particles,batch_size,observation_space,action_space,hidden_sizes = (256,256),
                activation=nn.ReLU):
        super().__init__()
        self.noise_dim = noise_dim
        self.n_particles = n_particles
        self.batch_size = batch_size
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]

        self.hidden_sizes = hidden_sizes
        self.activation = activation

        # self.state_layer = mlp([self.obs_dim, hidden_sizes[0]], activation) 
        # self.noise_layer = mlp([self.act_dim, hidden_sizes[0]],activation)
        # self.layers = mlp(list(hidden_sizes) + [self.act_dim], activation, nn.Tanh)

        self.concat = mlp([self.act_dim + self.noise_dim] + list(hidden_sizes),activation)
        self.layer2 =  mlp(list(hidden_sizes) + [self.act_dim], nn.Tanh, nn.Tanh)

    def _forward(self,state,n_particles = 1):
        n_state_samples = state.shape[0]

        if n_particles > 1:
            state = state.unsqueeze(1)
            state = state.repeat(1,n_particles,1)

            assert state.dim() == 3
            latent_shape = (n_state_samples, n_particles,
                            self.act_dim)
        else:
            latent_shape = (n_state_samples, self.act_dim)

        noise = T.rand(latent_shape)*4-2
        # state_out = self.state_layer(state)
        # noise_out = self.noise_layer(noise)
        #print("noise_out.shape=",noise_out.shape,"state_out.shape=",state_out.shape)
        #tmp = state_out.unsqueeze(-1)
        #print("before tmp.shape=",tmp.shape)
        #tmp = tmp + noise_out
        #print("after tmp.shape=",tmp.shape)
        samples = self.concat(T.cat([state, noise],dim=-1))
        #print("samples.shape=",samples.shape)
        drop = nn.Dropout(0.1)
        samples = self.layer2(drop(samples))
        return T.tanh(samples) if n_state_samples > 1 else T.tanh(samples).squeeze(0)

    def forward(self,state,n_particles=1):
        return self._forward(state,n_particles=n_particles)

    def act(self,state,n_particles=1):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with T.no_grad():
            return self._forward(state,n_particles).numpy()

# class ActionValueNetwork(nn.Module):
#     def __init__(self, lr, state_dim, action_dim, n_particles,
#             name='ActionValueNetwork', chkpt_dir='tmp/sql'):
#         super(ActionValueNetwork, self).__init__()
        
#         # Dimentions
#         self.state_dim = state_dim
#         self.n_particles = n_particles
#         self.action_dim = action_dim
#         self.name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sql')


#         # Initialize Input dimentions
#         self.state_dim = self.state_dim
#         self.hidden_dim = 512
#         self.output_dim = 1
        
#         #self.fc1 = nn.Linear(self.fc1_dim, self.output_dim)
        
#         # Define the NN layers
#         self.state = nn.Linear(self.state_dim, self.hidden_dim)
#         self.action = nn.Linear(self.action_dim, self.hidden_dim)
#         self.hidden_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.hidden_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.output = nn.Linear(self.hidden_dim, self.output_dim)
        
#         # self.bn1 = nn.BatchNorm1d(self.hidden_dim)
#         # self.bn2 = nn.BatchNorm1d(self.hidden_dim)
#         # self.bn3 = nn.BatchNorm1d(self.output_dim)
        
#         #self.apply(self.init_weights)
       
#         self.double()
        
        
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

#         self.to(self.device)

#     def forward(self, state, action):
#         X_s = self.state(state)
#         X_a = self.action(action)
#         X = T.relu(X_s + X_a)
#         X = T.relu(self.hidden_1(X))
#         #X = T.relu(self.hidden_2(X))
#         X = self.output(X)
#         return X
    
#     def init_weights(self, m):
#         if type(m) == nn.Linear:
#             nn.init.xavier_normal_(m.weight)
#     def save_checkpoint(self):
#         T.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))




# class SamplerNetwork(nn.Module):
#     def __init__(self, lr, state_dim, n_particles,
#             action_dim=2, max_action=1000, name='Sampler', chkpt_dir='tmp/sql'):
#         super(SamplerNetwork, self).__init__()
        
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.n_particles = n_particles
#         self.max_action = max_action


#         # Initialize Input dimentions
#         self.state_dim = self.state_dim
#         self.noise_dim = self.action_dim
#         self.hidden_dim = 128
#         self.output_dim = self.action_dim
        
        
#         self.state = nn.Linear(self.state_dim, self.hidden_dim)
#         self.noise = nn.Linear(self.noise_dim, self.hidden_dim)        
#         self.hidden_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.hidden_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.hidden_3 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.output = nn.Linear(self.hidden_dim, self.output_dim)
        
#         #self.apply(self.init_weights)
#         # self.bn1 = nn.BatchNorm1d(self.hidden_dim)
#         # self.bn2 = nn.BatchNorm1d(self.hidden_dim)
#         # self.bn3 = nn.BatchNorm1d(self.hidden_dim)
#         # self.bn4 = nn.BatchNorm1d(self.hidden_dim)
    
#         self.double()
        
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)
#         self.name = name
#         self.checkpoint_dir = chkpt_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_Sampler')

#     def forward(self, state, noise):
#         X_s = self.state(state)
#         X_n = self.noise(noise)
#         X = T.tanh(X_s + X_n)
#         X = self.hidden_1(X)
#         X = T.tanh(X)
#         X = self.hidden_2(X)
#         X = T.tanh(X)
#         X = self.hidden_3(X)
#         X = T.tanh(X)
#         X = self.output(X)
#         #return F.tanh(X) * T.tensor(self.max_action).to(self.device)
#         return T.tanh(X)
    
#     # def forward(self, state, noise):
#     #     X_s = self.state(state)
#     #     X_n = self.noise(noise)
#     #     X = F.dropout(T.tanh(X_s + X_n), 0.2)
#     #     X = self.hidden_1(X)
#     #     X = F.dropout(T.tanh(X), 0.2)
#     #     X = self.hidden_2(X)
#     #     X = F.dropout(T.tanh(X), 0.2)
#     #     X = self.hidden_3(X)
#     #     X = F.dropout(T.tanh(X), 0.2)
#     #     X = self.output(X)
#     #     #return F.tanh(X) * T.tensor(self.max_action).to(self.device)
#     #     return T.tanh(X)
    
#     # def forward(self, state, noise):
#     #     X_s = self.state(state)
#     #     X_n = self.noise(noise)
#     #     X = T.tanh(self.bn1(X_s + X_n))
#     #     X = self.fc2(X)
#     #     X = T.tanh(self.bn2(X))
#     #     X = self.fc3(X)
#     #     X = T.tanh(self.bn3(X))
#     #     X = self.fc4(X)
#     #     X = T.tanh(self.bn4(X))
#     #     X = self.fc5(X)
#     #     print(self.max_action)
#     #     #return F.tanh(X) * T.tensor(self.max_action).to(self.device)
#     #     return F.tanh(X)
    
#     def init_weights(self, m):
#         if type(m) == nn.Linear:
#             nn.init.xavier_normal_(m.weight)

#     def save_checkpoint(self):
#         T.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))





