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
  #train deepq agent on env
  print("Initial State: "+str((env.initial_state).T))
  print("Goal State: "+str((env.goal).T))
  #print("Max_reward: "+str(env.reward_max))
  print("State Space Element Type/Shape:", type(env.goal), env.goal.shape)
  #agent has 1 mlp hidden layer with 256 units
  mlp= models.mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False)
  
  # DQN
  # act = deepq.learn(env,mlp,lr=1e-3,total_timesteps=80000*env.n,buffer_size=1000000,exploration_fraction=0.05,
  #     exploration_final_eps=0.01,train_freq=1,batch_size=128,gamma=1,
  #     print_freq=200,checkpoint_freq=1000,target_network_update_freq=16*env.n,callback=callback, dueling=False)
  
  # Dueling DQN
  act = deepq.learn(env,mlp,lr=1e-3,total_timesteps=80000*env.n,buffer_size=1000000,exploration_fraction=0.05,
      exploration_final_eps=0.01,train_freq=1,batch_size=128,gamma=1,
      print_freq=200,checkpoint_freq=1000,target_network_update_freq=16*env.n,callback=callback)
  
  #save trained model 
  print("Saving model to "+save_path)
  act.save_act(save_path)
  

def test(env,load_path,num_episodes=1000):
  act = deepq.load_act(load_path+".pkl")
  # success_count=0.0
  test_render_file = open(load_path+".txt","w")
  best_obs = np.ones(env.n * env.m, dtype=int)
  best_episode_rew = -1*env.n
  for i in range(num_episodes):
      obs, done = env.reset(), False
      episode_rew = 0.0
      while not done:
          render_string = env.render(mode='ansi')+"\n"
          test_render_file.write(render_string)  
          obs, rew, done, _ = env.step(act(obs[None])[0])
          episode_rew += rew
      if episode_rew < best_episode_rew:
        best_episode_rew = episode_rew
        best_obs = obs 
      render_string = env.render(mode='ansi')+"\n"
      test_render_file.write(render_string)
      test_render_file.write("Episode reward "+str(episode_rew)+"\n")
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
        filename = "dq_bitflip"+str(n)+"_"+str(m)+"_"+str(space_seed)
        with tf.Graph().as_default():
            train(env,save_path+filename+".pkl")
        with tf.Graph().as_default():
            test(env,save_path+filename,num_episodes) 
            test_results_file.write("Seq Len:"+str(n)+","+" Num Seq:"+str(m)+","+" Seed:"+str(space_seed)+"\n")
  test_results_file.close()

if __name__=="__main__":
    main()
    # main2()