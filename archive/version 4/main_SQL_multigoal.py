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
from networks import MLPActorCritic, SamplingNetwork


# env= MultiGoalEnv()
# hid = 256
# l =2
# epochs=20
# seed = 42
# gamma =0.99

# ac,ac_targ,sampler = svgd(lambda : gym.make(env) , actor_critic=MLPActorCritic, sampler=SamplingNetwork, ac_kwargs=dict(hidden_sizes=[hid]*l),
#          steps_per_epoch=100, replay_size=int(1e6),
#          polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100,noise_dim = 2, n_particles=16, start_steps=0, 
#          update_after=1000, update_every=1, act_noise=0.1, num_test_episodes=10, 
#          max_ep_len=30, save_freq=1, gamma=gamma, seed=seed, epochs=epochs)

# lambda : gym.make(env) , actor_critic=MLPActorCritic, ac_kwargs=dict(hidden_sizes=[hid]*l),
#          steps_per_epoch=100, replay_size=int(1e6),
#          polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100,noise_dim = 2, n_particles=16, start_steps=0, 
#          update_after=1000, update_every=1, act_noise=0.1, num_test_episodes=10, 
#          max_ep_len=30, save_freq=1, gamma=gamma, seed=seed, epochs=epochs

if __name__ == "__main__":
    
    env= MultiGoalEnv()
    
    
    
    agent = Agent(lambda : gym.make(env) , actor_critic=MLPActorCritic, sampler=SamplingNetwork, hidden_dim=[256,256],
        replay_size=int(1e6), pi_lr=1e-3, q_lr=1e-3, batch_size=100, n_particles=16, gamma=0.99)
    
    epochs=20
    update_every=1
    update_after=50
    max_ep_len=30
    start_steps=0
    steps_per_epoch=100
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
        
        print(t)
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
    




# ac,ac_targ,sampler = svgd(lambda : gym.make(env) , actor_critic=MLPActorCritic, ac_kwargs=dict(hidden_sizes=[hid]*l), 
#      gamma=gamma, seed=seed, epochs=epochs)



# plotter = QFPolicyPlotter(qf = ac.q, policy=sampler, obs_lst=[[0,0],[-2.5,-2.5],[2.5,2.5],[-2.5,2.5],[2.5,-2.5]], default_action =[np.nan,np.nan], n_samples=100)
# plotter.draw()




# def plot(agent):
#     paths = []
#     actions_plot=[]
#     env = MultiGoalEnv()
#     n_games = 50
#     max_episode_length = 20
#     for i in range(n_games):
#         observation = env.reset()
#         episode_length = 0
#         done = False
#         score = 0
#         path = {'infos':{'pos':[]}}
#         while not done:
#             env.render()
#             #print('state: ', np.squeeze(observation))
#             action = agent.get_action_svgd(T.from_numpy(observation).unsqueeze(0).double()).squeeze().detach().numpy()

#             #print('ac: ', action[0].cpu().detach().numpy())
#             observation_, reward, done, info = env.step(action)
#             path['infos']['pos'].append(observation)
            
#             if episode_length == max_episode_length:
#                 done = True
#             episode_length += 1
            
#             #print('re:', reward)
#             score += reward
#             observation = observation_
#         paths.append(path)
        
#         score = score / 200
#         score_history.append(score)
#         avg_score = np.mean(score_history[-20:])
        
#     env.render_rollouts(paths, fout="test_%d.png" % i)


# if __name__ == '__main__':
#     env = MultiGoalEnv()
#     # print(env.observation_space.shape)
#     # print(env.action_space.shape)
#     agent = Agent(state_dim=env.observation_space.shape[0], env=env,
#             action_dim=env.action_space.shape[0], n_particles=32, batch_size=100, reward_scale=0.1, max_action=env.action_space.high)
#     n_games = 50
#     # uncomment this line and do a mkdir tmp && mkdir video if you want to
#     # record video of the agent playing the game.
#     #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
#     filename = 'inverted_pendulum.png'
#     figure_file = 'plots/' + filename

#     #print(env.action_space.high)
    
#     best_score = env.reward_range[0]
#     score_history = []
#     load_checkpoint = False
    
#     max_episode_length = 100

#     if load_checkpoint:
#         agent.load_models()
#         env.render(mode='human')

#     for i in range(n_games):
        
#         #observation = env.reset(init_state=[2.5, 2.5])
#         observation = env.reset()
#         episode_length = 0
        
#         done = False
#         score = 0
        
#         while not done:
#             #env.render()
#             #print('state: ', np.squeeze(observation))
#             action = agent.get_action_svgd(T.from_numpy(observation).unsqueeze(0)).squeeze().detach().numpy()
            
           
#             #print('ac: ', np.squeeze(action))
#             observation_, reward, done, info = env.step(action)
            
#             if episode_length == max_episode_length:
#                 done = True
#             #print(episode_length)
#             #print('re:', reward)
#             #print('Q: ', np.squeeze(env.get_Q()))
#             score += reward
#             agent.remember(observation, action, reward, observation_, done)
#             if not load_checkpoint:
#                 agent.learn(episode_length)
#             observation = observation_
#             episode_length += 1

            
#         score = score 
#         score_history.append(score)
#         avg_score = np.mean(score_history[-20:])
        
#         if avg_score > best_score:
#             best_score = avg_score
#             # if not load_checkpoint:
#             #     agent.save_models()
#         print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
        
#         plot(agent)
        
        
        
#     #agent.actor.sample_normal(T.FloatTensor([2.5,2.5]).repeat([100,1]))[0].detach().numpy()
    
#     plotter = QFPolicyPlotter(qf = agent.Q_Network, agent=agent, obs_lst=[[-2.5,-2.5],[0,0],[2.5,2.5]], default_action =[np.nan,np.nan], n_samples=50)
#     plotter.draw()


#     env.close()  
    
    

    
    
