# pybullet_envs
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

def plot(agent):
    paths = []
    actions_plot=[]
    env = MultiGoalEnv()
    n_games = 50
    max_episode_length = 20
    for i in range(n_games):
        observation = env.reset()
        episode_length = 0
        done = False
        score = 0
        path = {'infos':{'pos':[]}}
        while not done:
            env.render()
            #print('state: ', np.squeeze(observation))
            action = agent.choose_action_svgd(np.expand_dims(observation, axis=0))
            state = T.tensor(observation, dtype=T.float)
            
            agent.Q_Network.eval()
            Q_values = agent.get_Q_value(state, action, particles=True)[1]
            Q_max = T.argmax(Q_values)
            action = action[Q_max, :].detach().numpy()
            #print('ac: ', action[0].cpu().detach().numpy())
            observation_, reward, done, info = env.step(action)
            path['infos']['pos'].append(observation)
            
            if episode_length == max_episode_length:
                done = True
            episode_length += 1
            
            #print('re:', reward)
            score += reward
            observation = observation_
        paths.append(path)
        
        score = score / 200
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])
        
    env.render_rollouts(paths, fout="test_%d.png" % i)


if __name__ == '__main__':
    env = MultiGoalEnv()
    # print(env.observation_space.shape)
    # print(env.action_space.shape)
    agent = Agent(state_dim=env.observation_space.shape[0], env=env,
            action_dim=env.action_space.shape[0], n_particles=32, max_action=env.action_space.high)
    n_games = 50
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename

    #print(env.action_space.high)
    
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    
    max_episode_length = 400

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        
        #observation = env.reset(init_state=[2.5, 2.5])
        observation = env.reset()
        episode_length = 0
        
        done = False
        score = 0
        while not done:
            env.render()
            #print('state: ', np.squeeze(observation))
            action = agent.choose_action_svgd(np.expand_dims(observation, axis=0))
            state = T.tensor(observation, dtype=T.float)
            
            agent.Q_Network.eval()
            Q_values = agent.get_Q_value(state, action, particles=True)[1]
            Q_max = T.argmax(Q_values)
            action = action[Q_max, :].detach().numpy()
            #print('ac: ', np.squeeze(action))
            observation_, reward, done, info = env.step(action)
            
            if episode_length == max_episode_length:
                done = True
            episode_length += 1
            
            #print('re:', reward)
            #print('Q: ', np.squeeze(env.get_Q()))
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            
            
        score = score / 200
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])
        
        if avg_score > best_score:
            best_score = avg_score
            # if not load_checkpoint:
            #     agent.save_models()
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
        
        plot(agent)
        
        
    #agent.actor.sample_normal(T.FloatTensor([2.5,2.5]).repeat([100,1]))[0].detach().numpy()
    
    plotter = QFPolicyPlotter(qf = agent.critic_1, policy=agent.actor, obs_lst=[[-2,0],[0,2],[2.5,2.5]], default_action =[np.nan,np.nan], n_samples=50)
    plotter.draw()


    env.close()  
    
    

    
    


    
    
    
    # paths = []
    # actions_plot=[]
    # env = MultiGoalEnv()
    
    # for i in range(n_games):
        
    #     observation = env.reset()
    #     episode_length = 0
    #     done = False
    #     score = 0
    #     path = {'infos':{'pos':[]}}
    #     while not done:
    #         env.render()
    #         #print('state: ', np.squeeze(observation))
    #         action = agent.choose_action(observation)
    #         #print('ac: ', np.squeeze(action))
    #         observation_, reward, done, info = env.step(action)
    #         path['infos']['pos'].append(observation)
            
    #         if episode_length == max_episode_length:
    #             done = True
    #         episode_length += 1
            
    #         #print('re:', reward)
    #         score += reward
    #         if not load_checkpoint:
    #             agent.learn()
    #         observation = observation_
    #     paths.append(path)
        
    #     score = score / 200
    #     score_history.append(score)
    #     avg_score = np.mean(score_history[-20:])
        
    #     env.render_rollouts(paths)


####################################################
#     for episode in range(n_particles):
#       observation = env.reset()
#       done = False
#       step = 0
#       path = {'infos':{'pos':[]}}
#       particles = None
#       while not done and step < 20:
#         #a = get_action(observation,0)
#         #print("a.shape=",a.shape)
#         actions = get_sample(observation)
#         # a = b[0]
#         #print("before b.shape=",b.shape, "a.shape=",a.shape)
# #             print("a=",a)
#         # if (observation[0]<=2.6) & (observation[0]>=2.4)& (observation[1]>=2.4)&(observation[1]<=2.6):
#         #    actions_plot.append(a)
#         # print(f'actions.shape={actions.shape}')

#         observation, reward, done, _ = env.step(actions)
#         path['infos']['pos'].append(observation)
#         # print(f'observation.shape={observation.shape}')
#         step +=1
#       paths.append(path)
#     print("saving figure..., epoch=",epoch)
# #         with open('./actions_'+str(epoch)+'.txt', 'w') as filehandle:
# #             for listitem in actions_plot:
# #                 filehandle.write('%s\n' % listitem) 
#     env.render_rollouts(paths,fout="test_%d.png" % epoch)
 



