import gym
import numpy as np
import pybulletgym
from SQL_torch import Agent
from utils import plot_learning_curve
from gym import wrappers
import math
from multigoal import MultiGoalEnv
import torch as T
from plotter import QFPolicyPlotter
from networks import SamplingNetwork


# import gym
# env = gym.make('Walker2d-v2')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()


if __name__ == "__main__":
    
    #env= MultiGoalEnv()
    env = gym.make('Walker2d-v2')
    print(env.observation_space.shape)
    print(env.action_space.shape)
    agent = Agent(env, 
                  hidden_dim=[128, 128], replay_size=int(1e6), pi_lr=3e-4, 
                  q_lr=3e-4, batch_size=128, n_particles=16, gamma=0.99)
    
    epochs=15
    update_every=1
    update_after=2000
    max_ep_len=30
    start_steps=0
    steps_per_epoch=1000
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    
    
    
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    average_reward_data = []
    reward_data = []
    
    for t in range(total_steps):
        #env.render()
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = agent.get_sample(o)
        else:
            a = env.action_space.sample()
        
        # Step the env
        o2, r, d, _ = env.step(a)
        reward_data.append(r)
        print(t)
        ep_ret += r
        ep_len += 1
        #print("t=",t,"ep_ret=",ep_ret, "ep_len=",ep_len)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        agent.replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            #logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
            average_reward_data.append(reward_data)
            reward_data = []
            
            epoch = (t+1) // steps_per_epoch
            print("epoch=",epoch)
        
        # Update handling
        if t >= update_after and t % update_every == 0:
            batch = agent.replay_buffer.sample_batch(agent.batch_size)
            #print("before updating..")
            agent.learn(data=batch)
            #print("after updating..")

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            print("epoch=",epoch)
            #agent.plot_paths(epoch)

    average_reward_data = np.array(average_reward_data)        
    np.save('reward_walker', average_reward_data)

    # plotter = QFPolicyPlotter(qf = agent.Q_Network, ss=agent.SVGD_Network, obs_lst=[[0,0],[-2.5,-2.5],[2.5,2.5],[-2.5,2.5],[2.5,-2.5]], default_action =[np.nan,np.nan], n_samples=100)
    # plotter.draw()
    


