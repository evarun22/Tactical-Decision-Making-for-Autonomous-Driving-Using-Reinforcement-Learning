import gym
import csv
from gym import wrappers
import numpy as np
from dueling_ddqn_agent import DuelingDDQNAgent

import highway_env
import gym.spaces

if __name__ == '__main__':
    env = gym.make('highway-v0')
    best_score = -np.inf
    best_score_test = -np.inf
    load_checkpoint = True 
    evaluate = True
    idx_l=0
    n_games = 1000000
    mem_size= 100000
    mem_fill_initial = 1
    env = gym.make("highway-v0")
    screen_width, screen_height = 150, 150
    config = {
    "offscreen_rendering": False, #change for colab to True
    "observation": {
        "type": "GrayscaleObservation",
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "stack_size": 4,
        "observation_shape": (screen_width, screen_height)
        
    },
    "screen_width": screen_width,
    "screen_height": screen_height,
    "scaling": 2.5,
    
    "policy_frequency": 10, #controls the frequency at which decisions are taken, and thus that at which transitions (s,a,s') are observed
    "vehicles_count": 20,
    "duration":3000,
    "lanes_count":4,
    "vehicles_density":1,
    "simulation_frequency":10, #controls the frequency at which the world is refreshed. Increasing it will increase the accuracy of the simulation, but also the computation time.
    'show_trajectories': False,
    "action": {
        "type": "DiscreteMetaAction"
    }
    }

    env.configure(config)
    env.reset()
    if load_checkpoint:
        epsilon=0.01
    else:
        epsilon = 1
        
    if evaluate:
        epsilon=0.0
    else:
        epsilon = 1
      
    agent = DuelingDDQNAgent(gamma=0.99, epsilon=epsilon, lr=0.001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=mem_size, eps_min=0.1,
                     batch_size=128, replace=10000, eps_dec=5e-6,
                     chkpt_dir='models/', algo='DuelingDDQNAgent',
                     env_name='highway-v0')

    if load_checkpoint:
        agent.load_models()

    

    n_steps = 0
    steps_array,scores, eps_history, episode_array,loss_l = [], [], [],[], []
    
    print('Filling Replay Buffer with %d steps' %(mem_size))
    
    
    n_mem_steps=0
    
    if not evaluate:
        while n_mem_steps<mem_fill_initial:        
            done = False
            observation=env.reset()

            while not done:
                if not load_checkpoint:
                    action = env.action_space.sample()
                else:                
                    action = agent.choose_action(observation)

                observation_, reward, done, info = env.step(action)            


                agent.store_transition(observation, action,
                                         reward, observation_, int(done))

                observation = observation_
                n_mem_steps+=1
            if n_mem_steps%100==0:
                  print(100*(n_mem_steps/mem_fill_initial),'%')
        

    for i in range(n_games):
        done = False
        
        observation = env.reset()
        
        score = 0
        while not done:
            
            action = agent.choose_action(observation)
            
            
            observation_, reward, done, info = env.step(action)
            
            score += reward

            if not evaluate:
                agent.store_transition(observation, action,reward, observation_, int(done))
            
            
                loss=agent.learn()
            else:
                env.render()
            n_steps += 1
            observation = observation_

        '''if (i+1)%2==0:
            score_test=0
            done_test = False
            observation_test = env.reset()
            while not done_test:
            
                action_test = agent.choose_action(observation_test,test=True)
                next_observation_test, reward_test, done_test, info_test = env.step(action_test)
                score_test+=reward_test  
                observation_test = next_observation_test
                env.render()
            if score_test>best_score_test:
                agent.save_models_test(score_test)
                best_score_test = score_test'''
            
            
            
        scores.append(score)
        steps_array.append(i)  
        if not evaluate:
            loss_l.append(loss)
        avg_score = np.mean(scores[-100:])
        if not evaluate:
            print('episode: ', i,'score: ', score,
                 ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
                'epsilon %.2f' % agent.epsilon, 'steps', n_steps,'loss', loss)
        else:
            print('episode: ', i,'score: ', score,
                 ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
                'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if i>20: #wait for 20 eps to set best score
          if not evaluate:
              if avg_score > best_score:
                  #if not load_checkpoint:
                  agent.save_models()
                  best_score = avg_score

        eps_history.append(agent.epsilon)
        episode_array.append(i)
        #save stats 
        if not evaluate:
            if (i)%100==0:
                    print('saving stats...')
                    with open('/content/drive/MyDrive/OpenAI_Highway/plots/stats.csv', 'a+') as csvfile:
                        try:
                          stat=list(zip(episode_array[idx_l:],scores[idx_l:],eps_history[idx_l:],loss_l[idx_l:],steps_array[idx_l:]))
                          writer = csv.writer(csvfile)      

                          for item in stat:

                              writer.writerow(item)
                          idx_l = len(steps_array)
                        except Exception as e:
                          print(e)
                          pass