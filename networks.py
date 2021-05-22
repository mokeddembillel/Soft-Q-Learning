import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np



class ActionValueNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, n_particles, fc1_dims=256, fc2_dims=256,
            name='ActionValueNetwork', chkpt_dir='tmp/sql'):
        super(ActionValueNetwork, self).__init__()
        
        # Dimentions
        self.state_dim = state_dim
        self.n_particles = n_particles
        self.action_dim = action_dim
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sql')


        # Initialize Input dimentions
        self.state_dim = self.state_dim
        self.fc2_dim = 128
        self.fc3_dim = 1
        
        #self.fc1 = nn.Linear(self.fc1_dim, self.fc3_dim)
        
        # Define the NN layers
        self.state = nn.Linear(self.state_dim, self.fc2_dim)
        self.action = nn.Linear(self.action_dim, self.fc2_dim)
        self.fc2 = nn.Linear(self.fc2_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.fc3_dim)
        
        # self.bn1 = nn.BatchNorm1d(self.fc2_dim)
        # self.bn2 = nn.BatchNorm1d(self.fc2_dim)
        # self.bn3 = nn.BatchNorm1d(self.fc3_dim)
        
        self.apply(self.init_weights)
       
        self.double()
        
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        X_s = self.state(state)
        X_a = self.action(action)
        
        X = T.relu(X_s + X_a)
        X = self.fc2(X)
        X = T.relu(X)
        X = self.fc3(X)
        return T.sigmoid(X)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))




class SamplerNetwork(nn.Module):
    def __init__(self, alpha, state_dim, n_particles,
            action_dim=2, max_action=1000, name='Sampler', chkpt_dir='tmp/sql'):
        super(SamplerNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_particles = n_particles
        self.max_action = max_action


        # Initialize Input dimentions
        self.state_dim = self.state_dim
        self.noise_dim = self.action_dim
        self.fc2_dim = 512
        self.fc3_dim = 512
        self.fc4_dim = 512
        self.fc5_dim = self.action_dim
        
        
        self.state = nn.Linear(self.state_dim, self.fc2_dim)
        self.noise = nn.Linear(self.noise_dim, self.fc2_dim)        
        self.fc2 = nn.Linear(self.fc2_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.fc2_dim)
        self.fc4 = nn.Linear(self.fc2_dim, self.fc2_dim)
        self.fc5 = nn.Linear(self.fc2_dim, self.fc5_dim)
        
        self.apply(self.init_weights)
        # self.bn1 = nn.BatchNorm1d(self.fc2_dim)
        # self.bn2 = nn.BatchNorm1d(self.fc2_dim)
        # self.bn3 = nn.BatchNorm1d(self.fc2_dim)
        # self.bn4 = nn.BatchNorm1d(self.fc2_dim)
    
        self.double()
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_Sampler')

    def forward(self, state, noise):
        X_s = self.state(state)
        X_n = self.noise(noise)
        X = F.dropout(T.tanh(X_s + X_n), 0.2)
        X = self.fc2(X)
        X = F.dropout(T.tanh(X), 0.2)
        X = self.fc3(X)
        X = F.dropout(T.tanh(X), 0.2)
        X = self.fc4(X)
        X = F.dropout(T.tanh(X), 0.2)
        X = self.fc5(X)
        #return F.tanh(X) * T.tensor(self.max_action).to(self.device)
        return T.tanh(X)
    
    # def forward(self, state, noise):
    #     X_s = self.state(state)
    #     X_n = self.noise(noise)
    #     X = T.tanh(self.bn1(X_s + X_n))
    #     X = self.fc2(X)
    #     X = T.tanh(self.bn2(X))
    #     X = self.fc3(X)
    #     X = T.tanh(self.bn3(X))
    #     X = self.fc4(X)
    #     X = T.tanh(self.bn4(X))
    #     X = self.fc5(X)
    #     print(self.max_action)
    #     #return F.tanh(X) * T.tensor(self.max_action).to(self.device)
    #     return F.tanh(X)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))





