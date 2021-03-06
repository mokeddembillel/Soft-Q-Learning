# -*- coding: utf-8 -*-
"""ddpg_svgd_pytorch_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Qc5XLfkg-wn3MN6MdiI8aDho4u1RitTq
"""

from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from gym.spaces import Box

#import core as core

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from gym.utils import EzPickle
from gym import spaces
#from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import Env

import scipy.signal
import torch.nn as nn

class MultiGoalEnv(Env, EzPickle):
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """
    def __init__(self,
                 goal_reward=10,
                 actuation_cost_coeff=30.0,
                 distance_cost_coeff=1.0,
                 init_sigma=0.1):
        EzPickle.__init__(**locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.zeros(2, dtype=np.float32)
        self.init_sigma = init_sigma
        self.goal_positions = np.array(
            (
                (5, 0),
                (-5, 0),
                (0, 5),
                (0, -5)
            ),
            dtype=np.float32)
        
        # self.goal_positions = np.array(
        #     (   (0,-5),
        #         (-2.5,-2.5),
        #         (-5,0),
        #         (-2.5,2.5),
        #         (0,5),
        #         (2.5,2.5),
        #         (5,0),
        #         (2.5,-2.5)       
        #     ),
        #     dtype=np.float32)
        self.goal_threshold = 0.05
        self.goal_reward = goal_reward
        self.action_cost_coeff = actuation_cost_coeff
        self.distance_cost_coeff = distance_cost_coeff
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.
        self.reset()
        self.observation = None

        self._ax = None
        self._env_lines = []
        self.fixed_plots = None
        self.dynamic_plots = []

    def reset(self):
        unclipped_observation = (
            self.init_mu
            + self.init_sigma
            * np.random.normal(size=self.dynamics.s_dim)
            )
        self.observation = np.clip(
            unclipped_observation,
            self.observation_space.low,
            self.observation_space.high)
        return self.observation

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            dtype=np.float32,
            shape=None)

    @property
    def action_space(self):
        return spaces.Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim, ),
            dtype=np.float32)

    def get_current_obs(self):
        return np.copy(self.observation)

    def step(self, action):
        action = action.ravel()

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high).ravel()

        observation = self.dynamics.forward(self.observation, action)
        observation = np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high)

        reward = self.compute_reward(observation, action)
        dist_to_goal = np.amin([
            np.linalg.norm(observation - goal_position)
            for goal_position in self.goal_positions
        ])
        done = dist_to_goal < self.goal_threshold
        if done:
            reward += self.goal_reward

        self.observation = np.copy(observation)

        return observation, reward, done, {'pos': observation}

    def _init_plot(self):
        fig_env = plt.figure(figsize=(7, 7))
        self._ax = fig_env.add_subplot(111)
        self._ax.axis('equal')

        self._env_lines = []
        self._ax.set_xlim((-7, 7))
        self._ax.set_ylim((-7, 7))

        self._ax.set_title('Multigoal Environment')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')

        self._plot_position_cost(self._ax)
    
    '''
    def render_rollouts(self, paths=()):
        """Render for rendering the past rollouts of the environment."""
        if self._ax is None:
            self._init_plot()

        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = []

        for path in paths:
            positions = np.stack(path['infos']['pos'])
            xx = positions[:, 0]
            yy = positions[:, 1]
            self._env_lines += self._ax.plot(xx, yy, 'b')

        plt.draw()
        plt.pause(0.01)
    '''
    
    def render_rollouts(self, paths=(),fout=None):
        """Render for rendering the past rollouts of the environment."""
        if self._ax is None:
            self._init_plot()
 
        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = []
 
        for path in paths:
            positions = np.stack(path['infos']['pos'])
            xx = positions[:, 0]
            yy = positions[:, 1]
            self._env_lines += self._ax.plot(xx, yy, 'b' if not 'color' in path['infos'] else path['infos']['color'])
       
        #if fout:
        print("fout rollout=",fout)
        #plt.savefig(fout)
 
        #plt.draw()
        plt.pause(0.01)

    def render(self, mode='human', *args, **kwargs):
        """Render for rendering the current state of the environment."""
        pass

    def compute_reward(self, observation, action):
        # penalize the L2 norm of acceleration
        # noinspection PyTypeChecker
        action_cost = np.sum(action ** 2) * self.action_cost_coeff

        # penalize squared dist to goal
        cur_position = observation
        # noinspection PyTypeChecker
        goal_cost = self.distance_cost_coeff * np.amin([
            np.sum((cur_position - goal_position) ** 2)
            for goal_position in self.goal_positions
        ])

        # penalize staying with the log barriers
        costs = [action_cost, goal_cost]
        reward = -np.sum(costs)
        return reward

    def _plot_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        goal_costs = np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)
        costs = goal_costs

        contours = ax.contour(X, Y, costs,20)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goal_positions[:, 0],
                       self.goal_positions[:, 1], 'ro')
        return [contours, goal]


class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
            np.random.normal(size=self.s_dim)
        return state_next

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.act_dim = act_dim
        # self.state_layer = nn.Linear(obs_dim,hidden_sizes[0])
        # self.action_layer = nn.Linear(act_dim,hidden_sizes[0])
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act,n_sample = 1):
        if n_sample > 1:
            obs = obs.unsqueeze(1).repeat(1,n_sample,1)
            assert obs.dim() == 3
        q = self.q(torch.cat([act,obs], dim=-1))
        return  torch.squeeze(q, -1).squeeze(-1)   # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

class SamplingNetwork(nn.Module):
    def __init__(self,noise_dim,n_particles,batch_size,observation_space,action_space,hidden_sizes = (256,256),
                activation=nn.Tanh):
        super().__init__()
        self.noise_dim = noise_dim
        self.n_particles = n_particles
        self.batch_size = batch_size
        self.obs_dim = observation_space.shape[0]
        self.act_dim = action_space.shape[0]

        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.state_layer = mlp([self.obs_dim, hidden_sizes[0],n_particles], activation) 
        self.noise_layer = mlp([noise_dim, hidden_sizes[0],hidden_sizes[0]],activation)
        self.layers = mlp([hidden_sizes[0]] + list(hidden_sizes) + [self.act_dim], activation, nn.Tanh)

        
    def _forward(self,state,noise):
        if noise is None:
            noise = torch.randn((self.n_particles,self.noise_dim))
        state_out = self.state_layer(state)
        noise_out = self.noise_layer(noise)
        #print("noise_out.shape=",noise_out.shape,"state_out.shape=",state_out.shape)
        #tmp = state_out.unsqueeze(-1)
        #print("before tmp.shape=",tmp.shape)
        #tmp = tmp + noise_out
        #print("after tmp.shape=",tmp.shape)
        samples = self.layers((state_out.unsqueeze(-1) + noise_out))
        #print("samples.shape=",samples.shape)
        return samples

    def forward(self,state,noise=None):
        return self._forward(state,noise)

    def act(self,state,noise=None):
         with torch.no_grad():
            return self._forward(state,noise).numpy()

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

        noise = torch.rand(latent_shape)*4-2
        # state_out = self.state_layer(state)
        # noise_out = self.noise_layer(noise)
        #print("noise_out.shape=",noise_out.shape,"state_out.shape=",state_out.shape)
        #tmp = state_out.unsqueeze(-1)
        #print("before tmp.shape=",tmp.shape)
        #tmp = tmp + noise_out
        #print("after tmp.shape=",tmp.shape)
        samples = self.concat(torch.cat([state, noise],dim=-1))
        #print("samples.shape=",samples.shape)
        drop = nn.Dropout(0.1)
        samples = self.layer2(drop(samples))
        return torch.tanh(samples) if n_state_samples > 1 else torch.tanh(samples).squeeze(0)

    def forward(self,state,n_particles=1):
        return self._forward(state,n_particles=n_particles)

    def act(self,state,n_particles=1):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            return self._forward(state,n_particles).numpy()

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
#         self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
#         self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
#         self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

def rbf_kernel(input_1, input_2,  h_min=1e-3):
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
    median_sq = torch.median(dist_sq, dim=1)[0]
    median_sq = median_sq.unsqueeze(1)
    # N * 1 * k_upd * 1

    h = median_sq / np.log(k_fix + 1.) + .001
    # N * 1 * k_upd * 1

    kappa = torch.exp(-dist_sq / h)
    # N * k_fix * k_upd * 1

    # Construct the gradient
    kappa_grad = -2. * diff / h * kappa
    return kappa, kappa_grad

# def svgd(env_fn, actor_critic=core.MLPActorCritic, sampler = core.SamplingNetwork,ac_kwargs=dict(), seed=0, 
#          steps_per_epoch=40, epochs=100, replay_size=int(1e6), gamma=0.99, 
#          polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100,noise_dim = 100, n_particles=100, start_steps=10000, 
#          update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10, 
#          max_ep_len=1000, save_freq=1):
         
def svgd(env_fn, actor_critic=MLPActorCritic, sampler = SamplingNetwork,ac_kwargs=dict(), seed=0, 
         steps_per_epoch=100, epochs=100, replay_size=int(1e6), gamma=0.99, 
         polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100,noise_dim = 2, n_particles=16, start_steps=0, 
         update_after=1000, update_every=1, act_noise=0.1, num_test_episodes=10, 
         max_ep_len=30, save_freq=1):
    #logger = EpochLogger(**logger_kwargs)
    #logger.save_config(locals())
    env= MultiGoalEnv()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    #env, test_env = env_fn(), env_fn()
    env, test_env = env, env
    obs_dim = env.observation_space.shape
    #print(f'env.action_space.shape ={env.action_space.shape }')
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    sampler = sampler(noise_dim = noise_dim, batch_size = batch_size, n_particles = n_particles,
                    observation_space= env.observation_space, action_space = env.action_space)
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    #var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.q])
    #logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        aplus = torch.from_numpy(sampler.act(o2,n_particles=n_particles))
        #print("aplus=",aplus.shape)
        
        q = ac.q(o,a)
        

        # Bellman backup for Q function
        with torch.no_grad():
            #q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            q_pi_targ = ac_targ.q(o2, aplus,n_sample=n_particles).mean(dim=1)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        print(f'loss_q: {loss_q}')
        print(f'q_pi_targ: {q_pi_targ}')
        print(f'q1: {q}')

        return loss_q, loss_info

    # Set up function for computing svgd pi loss
    def compute_loss_pi_svgd(data):
        o = data['obs']
        ss_targ = torch.from_numpy(sampler.act(o))
        ss_targ.requires_grad = False
        a = ac.pi(o)
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

    def compute_loss_pi(data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()


    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)
    sampler_optimizer = Adam(sampler.parameters(), lr=pi_lr)

    # Set up model saving
    #logger.setup_pytorch_saver(ac)


    def update_svgd_ss(data,train=True):
        o = data['obs']
        actions = sampler(o,n_particles=n_particles)
        assert actions.shape == (batch_size,n_particles,act_dim)
        # SVGD requires computing two empirical expectations over actions
        # (see Appendix C1.1.). To that end, we first sample a single set of
        # actions, and later split them into two sets: `fixed_actions` are used
        # to evaluate the expectation indexed by `j` and `updated_actions`
        # the expectation indexed by `i`.
        # n_fixed_actions = n_updated_actions = n_particles//2
        # fixed_actions, updated_actions = torch.split(actions, [n_fixed_actions, n_updated_actions], dim=1)
        # fixed_actions.detach()
        # fixed_actions.requires_grad = True
        fixed_actions = sampler.act(o,n_particles=n_particles)
        fixed_actions = torch.from_numpy(fixed_actions)
        fixed_actions.requires_grad = True
        svgd_target_values = ac.q(o, fixed_actions,n_sample = n_particles)

         # Target log-density. Q_soft in Equation 13:
        # squash_correction = torch.sum(
        #     torch.log(1 - fixed_actions**2 + 1e-6), dim=-1)
        # log_p = svgd_target_values + squash_correction
        log_p = svgd_target_values


        grad_log_p = torch.autograd.grad(log_p.sum(), fixed_actions)[0]
        grad_log_p = grad_log_p.view(batch_size,n_particles,act_dim).unsqueeze(2)
        grad_log_p = grad_log_p.detach()
        assert grad_log_p.shape == (batch_size, n_particles, 1, act_dim)

        kappa, gradient = rbf_kernel(input_1=fixed_actions, input_2=actions)

         # Kernel function in Equation 13:
        # kappa = kappa.unsqueeze(dim=3)
        assert kappa.shape == (batch_size, n_particles, n_particles, 1)

        # Stein Variational Gradient in Equation 13:
        T_C = total_steps/steps_per_epoch
        anneal = ((t%(T_C))/T_C)**2
        anneal = 1.
        action_gradients = (1/n_particles)*torch.sum(anneal*kappa * grad_log_p + gradient, dim=1)
        assert action_gradients.shape == (batch_size, n_particles, act_dim)

        # Propagate the gradient through the policy network (Equation 14).
        if train:
            sampler_optimizer.zero_grad()
            torch.autograd.backward(-actions,grad_tensors=action_gradients)
            #torch.nn.utils.clip_grad_norm_(g_net.parameters(),2)
            sampler_optimizer.step()

    def update(data):

        # update stein sampler 
        update_svgd_ss(data)
        #print("done update_svgd_ss")
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

  
        # Freeze SS-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        
        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
#         for p in ac.q.parameters():
#             p.requires_grad = False
        for p in sampler.parameters():
            p.requires_grad = False
            

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi,loss_pi_info = compute_loss_pi_svgd(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in sampler.parameters():
            p.requires_grad = True
            
        # Unfreeze Q-network so you can optimize it at next DDPG step.
#         for p in ac.q.parameters():
#             p.requires_grad = True

        # Record things
        #logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        return np.clip(a, -act_limit, act_limit)
    
     
    def get_sample(o,n_sample=1):
        a = sampler.act(torch.as_tensor(o, dtype=torch.float32),n_particles=n_sample)
        # a += act_noise * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit) 


    def plot_paths(epoch):
        paths = []
        actions_plot=[]
        env = MultiGoalEnv()

        for episode in range(n_particles):
          observation = env.reset()
          done = False
          step = 0
          path = {'infos':{'pos':[]}}
          particles = None
          while not done and step < max_ep_len:
            #a = get_action(observation,0)
            #print("a.shape=",a.shape)
            actions = get_sample(observation,1)
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


    def test_agent():
        all_ret = []
        all_len = []
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            all_ret.append(ep_ret)
            all_len.append(ep_len)

            #logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        # print(f'AvgTestEpRet={sum(ep_ret)/num_test_episodes}, AvgTestEpLen={sum(ep_len)/num_test_episodes}')
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_sample(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        #print("t=",t,"ep_ret=",ep_ret, "ep_len=",ep_len)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            #logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            batch = replay_buffer.sample_batch(batch_size)
            #print("before updating..")
            update(data=batch)
            #print("after updating..")

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            print("epoch=",epoch)
            plot_paths(epoch)

            # Save model
            #if (epoch % save_freq == 0) or (epoch == epochs):
                #logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()
    return (ac,ac_targ,sampler)

env= MultiGoalEnv()
hid = 256
l =2
epochs=20
seed = 42
gamma =0.99
# svgd(lambda : gym.make(env) , actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[hid]*l), 
#      gamma=gamma, seed=seed, epochs=epochs)

ac,ac_targ,sampler = svgd(lambda : gym.make(env) , actor_critic=MLPActorCritic, ac_kwargs=dict(hidden_sizes=[hid]*l), 
     gamma=gamma, seed=seed, epochs=epochs)
#, logger_kwargs=logger_kwargs)

import numpy as np
import matplotlib.pyplot as plt


class QFPolicyPlotter:
    def __init__(self, qf, policy, obs_lst, default_action, n_samples):
        self._qf = qf
        self._policy = policy
        self._obs_lst = obs_lst
        self._default_action = default_action
        self._n_samples = n_samples

        self._var_inds = np.where(np.isnan(default_action))[0]
        assert len(self._var_inds) == 2

        n_plots = len(obs_lst)

        x_size = 5 * n_plots
        y_size = 5

        fig = plt.figure(figsize=(x_size, y_size))
        self._ax_lst = []
        for i in range(n_plots):
            ax = fig.add_subplot(100 + n_plots * 10 + i + 1)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            ax.grid(True)
            self._ax_lst.append(ax)

        self._line_objects = list()

    def draw(self):
        # noinspection PyArgumentList
        [h.remove() for h in self._line_objects]
        self._line_objects = list()

        self._plot_level_curves()

        self._plot_action_samples()

        plt.draw()
        plt.pause(0.001)

    def _plot_level_curves(self):
        # Create mesh grid.
        xs = np.linspace(-1, 1, 50)
        ys = np.linspace(-1, 1, 50)
        xgrid, ygrid = np.meshgrid(xs, ys)
        N = len(xs)*len(ys)

        # Copy default values along the first axis and replace nans with
        # the mesh grid points.
        actions = np.tile(self._default_action, (N, 1))
        actions[:, self._var_inds[0]] = xgrid.ravel()
        actions[:, self._var_inds[1]] = ygrid.ravel()
        actions = torch.from_numpy(actions.astype(np.float32))
        for ax, obs in zip(self._ax_lst, self._obs_lst):
            obs = torch.FloatTensor(obs).repeat([actions.shape[0],1])
            with torch.no_grad():
                qs = self._qf(obs, actions).numpy()

            qs = qs.reshape(xgrid.shape)

            cs = ax.contour(xgrid, ygrid, qs, 20)
            self._line_objects += cs.collections
            self._line_objects += ax.clabel(
                cs, inline=1, fontsize=10, fmt='%.2f')

    def _plot_action_samples(self):
        for ax, obs in zip(self._ax_lst, self._obs_lst):
            with torch.no_grad():
                actions = self._policy(torch.FloatTensor(obs).repeat([self._n_samples,1])).numpy()
            x, y = actions[:, 0], actions[:, 1]
            ax.title.set_text(str(obs))
            self._line_objects += ax.plot(x, y, 'b*')



plotter = QFPolicyPlotter(qf = ac.q, policy=sampler, obs_lst=[[0,0],[-2.5,-2.5],[2.5,2.5],[-2.5,2.5],[2.5,-2.5]], default_action =[np.nan,np.nan], n_samples=100)
plotter.draw()



plotter_pi = QFPolicyPlotter(qf = ac.q, policy=ac.pi, obs_lst=[[-2,0],[0,2],[0,0]], default_action =[np.nan,np.nan], n_samples=1)
plotter_pi.draw()

