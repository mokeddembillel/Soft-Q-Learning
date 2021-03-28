import gym
import numpy as np
from SQL_torch import Agent
from utils import plot_learning_curve
from gym import wrappers
import math

if __name__ == '__main__':
    env = gym.make('InvertedPendulumPyBulletEnv-v0')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    
    n_games = 100
    filename = 'inverted_pendulum.png'
    figure_file = 'plots/' + filename
    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            #env.render()
            
            # Sample an action for st using f
            action = agent.choose_action_svgd_network(noise, state) # not finished
            
            # Sample next state from the environment
            observation_, reward, done, info = env.step(action)
            
            # Sum the rewards
            score += reward
            # Save the new experience in the replay memor
            agent.remember(observation, action, reward, observation_, done)
            
            # Learn
            if not load_checkpoint:
                agent.learn()
            
            
            observation = observation_
        
        
        
        
        
        
        # Unimportant for now
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
                
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
    env.close()        
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)








