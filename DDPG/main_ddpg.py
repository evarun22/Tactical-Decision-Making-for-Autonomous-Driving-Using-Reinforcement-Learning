import gym
import highway_env
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    config = {
        "offscreen_rendering": False,
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted"
        },
        "action": {
            "type": "ContinuousAction"
        },
        'duration': 3000,
        'offroad_terminal':True,
        'policy_frequency':10,
        'simulation_frequency':10,
        'vehicles_count':20,
    }
    env = gym.make('highway-v0')
    env.configure(config)
    observation = env.reset()
    observation=observation.reshape(observation.shape[0]*observation.shape[1],)
    print(observation.shape)
    print(env.action_space.sample())
    print(env.config)
    agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=observation.shape, tau=0.001,
                    batch_size=64, fc1_dims=400, fc2_dims=300, 
                    n_actions=env.action_space.shape[0])
    n_games = 1000
    test_agent = True
    load_checkpoint=False
    
    if test_agent:
        n_games=100
        load_checkpoint=True
    filename = 'Highway_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = 'plots/' + filename + '.png'

    if load_checkpoint:
            agent.load_models()
    
    best_score = env.reward_range[0]
    score_history = []
    for i in range(n_games):
        observation = env.reset()
        observation=observation.reshape(observation.shape[0]*observation.shape[1],)
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_ = observation_.reshape(observation_.shape[0]*observation_.shape[1],)
            
            if not test_agent:
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
            score += reward
            observation = observation_
            env.render()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if not test_agent:
            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score, 'best score %.1f' % best_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)




