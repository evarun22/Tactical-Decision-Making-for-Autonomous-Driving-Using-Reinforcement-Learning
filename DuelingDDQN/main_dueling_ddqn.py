import gym
import highway_env
from gym import wrappers
import numpy as np
from dueling_ddqn_agent import DuelingDDQNAgent
from utils import plot_learning_curve, make_env

if __name__ == '__main__':
    
    config = {
        "offscreen_rendering": False,
        "observation": {
            "type": "TimeToCollision",
            "horizon": 10
            },
        "action": {
            "type": "DiscreteMetaAction"
        },
        'duration': 3000,
        'offroad_terminal':True,
        'policy_frequency':5,
        'simulation_frequency':5,
        'vehicles_count':20,
        'reward_speed_range': [0, 50],
    }
    
    env=gym.make('highway-v0')
    env.configure(config)
    observation = env.reset()
    observation=observation.flatten()
    print(observation.shape)
    best_score = -np.inf
    load_checkpoint = True
    n_games = 20
    agent = DuelingDDQNAgent(gamma=0.99, epsilon=0.01, lr=0.0001,
                     input_dims=(observation.flatten().shape),
                     n_actions=env.action_space.n, mem_size=70000, eps_min=0.01,
                     batch_size=32, replace=10000, eps_dec=1e-6,
                     chkpt_dir='models/', algo='DuelingDDQNAgent',
                     env_name='highway-v0')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()
        observation=observation.flatten()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation_=observation_.flatten()
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, int(done))
                agent.learn()
            observation = observation_
            n_steps += 1
            env.render()
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if load_checkpoint and n_steps >= 18000:
            break

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
