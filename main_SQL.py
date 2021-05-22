# pybullet_envs
import gym
import numpy as np
import pybulletgym
from SQL_torch import Agent
from utils import plot_learning_curve
from gym import wrappers
import math
import gym_lqr
import torch as T

if __name__ == '__main__':
    #env = gym.make('gym_lqr:lqr-v0')
    env = gym.make('InvertedPendulum-v2')

    print(env.observation_space.shape)
    print(env.action_space.shape)
    
    agent = Agent(state_dim=env.observation_space.shape[0], env=env,
            action_dim=env.action_space.shape[0], n_particles=32, max_action=env.action_space.high)
    
    #print(env.action_space.high)
    
    
    n_games = 100
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename
    best_score = env.reward_range[0]
    score_history = []
    state_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        #observation = env.reset(init_x=np.array([50]), max_steps=100)
        observation = env.reset()
        done = False
        score = 0
        state = 0
        while not done:
            env.render()
            
            # Sample an action for st using f
            #action = agent.choose_action_svgd(observation)[0].detach().numpy()
            action = agent.choose_action_svgd(np.expand_dims(observation, axis=0))
            state = T.tensor(np.expand_dims(observation, axis=0), dtype=T.float)
            
            agent.Q_Network.eval()
            Q_values = agent.get_Q_value(state, action)[1]
            Q_max = T.argmax(Q_values)
            action = action[0, Q_max].detach().numpy()
            #action = [np.random.choice(action.squeeze())]
            # print('st', observation.squeeze())
            # print('ac: ', action)
            # Sample next state from the environment
            observation_, reward, done, info = env.step(action)
            # Sum the rewards
            score += reward
            #state += observation
            # print('re:', reward)

            # Save the new experience in the replay memor
            agent.remember(observation, action, reward, observation_, done)
            
            #if not load_checkpoint:
            if not load_checkpoint:
                agent.learn()    
            
            observation = observation_
        
        # Unimportant for now
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        #state_history.append(state / 100)
#        avg_state = np.mean(state_history[-100:])

        
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
                
        #print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'avg_state %.1f' % avg_state)
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
    env.close()        
    # if not load_checkpoint:
    #     x = [i+1 for i in range(n_games)]
    #     plot_learning_curve(x, score_history, figure_file)








