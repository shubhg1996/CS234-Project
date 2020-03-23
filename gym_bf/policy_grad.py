import gym
import gym_bf
from baselines import deepq
from baselines.common import models
import numpy as np
from gym.envs.registration import register 
import os
import tensorflow as tf
from policy_grad_lib import PolicyGradient
import itertools

def callback(lcl, _glb):
    #for deepq training
    #stop training when mean reward for last 100 episodes <= (reward_max - reward_dist)
    # reward_dist = 0.0001
    #is_solved = (lcl['saved_mean_reward']!=None) and (lcl['saved_mean_reward']>=(lcl['env'].reward_max - reward_dist))
    is_solved = (len(lcl['episode_rewards']) > 25000)
    return is_solved
    
def make_env(n=10, m=2, space_seed=0):
  # create environment
  id = "BF"+str(n)+"_"+str(m)+"_"+str(space_seed)+"-v0"
  try :
    register(id=id,entry_point='gym_bf.envs:BFEnv',kwargs = {"space_seed":space_seed,"n":n, "m":m})
  except :
    print("Environment with id = "+id+" already registered.Continuing with that environment.")
  env=gym.make(id)
  env.seed(0)
  return env

def train(env,save_path):
  #train policy grad agent on env
  print("Initial State: "+str((env.initial_state).T))
  print("Goal State: "+str((env.goal).T))
  #print("Max_reward: "+str(env.reward_max))
  print("State Space Element Type/Shape:", type(env.goal), env.goal.shape)
  
  PG = PolicyGradient(
    n_x = env.observation_space.shape[0],
    n_y = env.action_space.n,
    learning_rate=0.001,
    reward_decay=1.,
    load_path=None,
    save_path=save_path
  )
  rewards = []
  for episode in range(10000):
    observation = env.reset()
    
    while True:
      action = PG.choose_action(observation)
      # action = np.random.randint(0, env.action_space.n)
      observation_, reward, done, info = env.step(action)
      PG.store_transition(observation, action, reward)
      if done:
        episode_rewards_sum = sum(PG.episode_rewards)
        rewards.append(episode_rewards_sum)
        max_reward_so_far = np.amax(rewards)
        if episode%10==0:
          print("==========================================")
          print("Episode: ", episode)
          print("Reward: ", episode_rewards_sum)
          print("Max reward so far: ", max_reward_so_far)
        PG.discount_and_norm_rewards()
        if episode%100==0:
          PG.learn()
        break
      observation = observation_

  #save trained model 
  print("Saving model to "+save_path)
  PG.saver.save(PG.sess, PG.save_path)
  PG.plot_cost()


def test(env,load_path,num_episodes=1000):
  # success_count=0.0
  test_render_file = open(load_path+".txt","w")
  best_obs = np.ones(env.n * env.m, dtype=int)
  best_episode_rew = -1*env.n
  
  rewards = []
  for episode in range(100):
    observation = env.reset()
    episode_rew = 0.0
    while True:
      action = PG.choose_action(observation)
      # action = np.random.randint(0, env.action_space.n)
      observation_, reward, done, info = env.step(action)
      episode_rew += reward
      env.render(mode='human')
      if done:
        episode_rewards_sum = episode_rew
        rewards.append(episode_rewards_sum)
        max_reward_so_far = np.amax(rewards)
        if episode%10==0:
          print("==========================================")
          print("Episode: ", episode)
          print("Reward: ", episode_rewards_sum)
          print("Max reward so far: ", max_reward_so_far)
        break
      observation = observation_
  
  test_render_file.close()
  obs_pm1 = best_obs*2 - 1
  state_pm1 = np.reshape(obs_pm1, [env.m, env.n])
  print('State')
  print(best_obs)
  print(state_pm1)
  print()
  pairs = np.array(list(itertools.combinations(range(env.m), 2)))
  first_i = pairs[:,0]; second_i = pairs[:,1]

  # Compute auto
  auto_corr_vectors = np.flip( np.fft.ifft(np.fft.fft(np.flip(state_pm1, axis=1)) * np.fft.fft(state_pm1)), axis=1 )
  mean_sqr_side_peak_auto = np.mean( np.square( np.abs(auto_corr_vectors[:, 1:]) ) )
  var_sqr_side_peak_auto = np.var( np.mean( np.square( np.abs(auto_corr_vectors[:, 1:]) ), axis=1 ) )
  var_sqr_side_peak_auto_norm = np.var( np.mean( np.square( np.abs(auto_corr_vectors[:, 1:]) ), axis=1 ) / (env.n*env.n) )
  print('Auto')
  print(np.real(auto_corr_vectors))
  print()

  # Compute average balance
  bal = np.mean( np.abs( np.sum(state_pm1, axis=1) ) )

  # Compute cross
  cross_corr_vectors = np.flip( np.fft.ifft(np.fft.fft(np.flip(state_pm1[first_i,:], axis=1)) * np.fft.fft(state_pm1[second_i,:])), axis=1 )
  mean_sqr_side_peak_cross = np.mean( np.square( np.abs(cross_corr_vectors) ) )
  var_sqr_side_peak_cross = np.var( np.mean( np.square( np.abs(cross_corr_vectors) ), axis=1 ) )
  var_sqr_side_peak_cross_norm = np.var( np.mean( np.square( np.abs(cross_corr_vectors) ), axis=1 ) / (env.n*env.n) )
  print('Cross')
  print(np.real(cross_corr_vectors))
  print()

  mean_sqr_side_peak = 0.5*mean_sqr_side_peak_auto + 0.5*mean_sqr_side_peak_cross

  print('Mean sqr (auto):', mean_sqr_side_peak_auto)
  print('Mean sqr (cross):', mean_sqr_side_peak_cross)
  print('Mean sqr:', mean_sqr_side_peak)
  print('Var sqr (auto):', var_sqr_side_peak_auto)
  print('Var sqr (cross):', var_sqr_side_peak_cross)
  print('Mean bal:', bal)

  print()
  print('----------Normalized----------')
  print('Mean sqr (auto):', mean_sqr_side_peak_auto / (env.n*env.n))
  print('Mean sqr (cross):', mean_sqr_side_peak_cross / (env.n*env.n))
  print('Mean sqr:', mean_sqr_side_peak / (env.n*env.n))
  print('Var sqr (auto):', var_sqr_side_peak_auto_norm)
  print('Var sqr (cross):', var_sqr_side_peak_cross_norm)
  print()
  

def main2(n=7, m=7, space_seed=12,num_episodes=100, save_path="./"):
  # For test only
  env = make_env(n,m,space_seed)
  filename = "pg_bitflip"+str(n)+"_"+str(m)+"_"+str(space_seed)
  test(env,save_path+filename,num_episodes)

def main(n_list=[5], m_list=[2],  space_seed_list=[12],num_episodes=1000,save_path="./"):
  test_results_file = open(save_path+"test_results.txt","w")
  for i in range(len(n_list)):
    n = n_list[i]
    m = m_list[i]
    for space_seed in space_seed_list:
        print("started for "+str(n)+","+str(space_seed))
        env = make_env(n,m,space_seed)
        filename = "pg_bitflip"+str(n)+"_"+str(m)+"_"+str(space_seed)
        with tf.Graph().as_default():
            train(env,save_path+filename+".pkl")
        with tf.Graph().as_default():
            test(env,save_path+filename,num_episodes) 
            test_results_file.write("Seq Len:"+str(n)+","+" Num Seq:"+str(m)+","+" Seed:"+str(space_seed)+"\n")
  test_results_file.close()

if __name__=="__main__":
    main()
    # main2()