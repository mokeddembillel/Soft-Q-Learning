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


if __name__ == "__main__":
    
    env= MultiGoalEnv()
    hid = 256
    l = 2
    
    agent = Agent(lambda : gym.make(env), 
                  hidden_dim=[256, 256], replay_size=int(1e6), pi_lr=1e-3, 
                  q_lr=1e-3, batch_size=500, noise_dim=2, n_particles=16, gamma=0.99)
    
    epochs=30
    update_every=1
    update_after=2000
    max_ep_len=30
    start_steps=0
    steps_per_epoch=400
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    
    
    
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = agent.get_sample(o)
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
        agent.replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            #logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0
        
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
            agent.plot_paths(epoch)

            # Save model
            #if (epoch % save_freq == 0) or (epoch == epochs):
                #logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            # agent.test_agent()
    plotter = QFPolicyPlotter(qf = agent.Q_Network, ss=agent.SVGD_Network, obs_lst=[[0,0],[-2.5,-2.5],[2.5,2.5],[-2.5,2.5],[2.5,-2.5]], default_action =[np.nan,np.nan], n_samples=100)
    plotter.draw()
    


