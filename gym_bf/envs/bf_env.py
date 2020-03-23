import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym import spaces
from gym.spaces.space import Space
import itertools
from numpy.fft import fft, ifft

def periodic_corr(x, y):
    """Periodic correlation, implemented using the FFT.

    x and y must be real sequences with the same length.
    """
    return ifft(fft(x) * fft(y).conj()).real

class BFEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, n=10, m=1, space_seed=0):
        self.n = n  # number bits per sequence
        self.m = m  # number sequences in family

        # there is an option NOT to flip any bit( index = n)
        self.action_space = spaces.Discrete( (self.n*self.m) + 1)
        # self.action_space = spaces.Discrete( (self.n) + 1)
        self.observation_space = spaces.MultiDiscrete([2]*self.n*self.m)
        # self.observation_space = spaces.MultiDiscrete([2]*self.n)
        pairs = np.array(list(itertools.combinations(range(m), 2)))
        self.first_i = pairs[:,0]; self.second_i = pairs[:,1]
        
        #self.reward_range = (-1, 0)
        self.reward_range = (-self.n, 0)  # if average, then worst max corr is -N and best is 0
        
        # self.space = Space()
        # self.space.seed(space_seed)
        self.space_seed = space_seed
        self.initial_state = self.observation_space.sample()
        # self.initial_state = np.ones((self.n))
        # print("Init_state: ",self.initial_state)
        
        # Can get rid of this later
        #self.goal = self.observation_space.sample()
        self.goal = np.zeros((self.n*self.m))

        # print("Goal: ",self.goal)
        self.state = self.initial_state
        self.envstepcount = 0
        self.seed()
        # self.reward_max = - \
        #     np.sum(np.bitwise_xor(self.initial_state, self.goal))+1
        # if(np.array_equal(self.goal, self.initial_state)):
        #     self.reward_max = 0
        
        #self.reward_max = 0 - np.sum(self.initial_state)
        self.update_reward_max()

    def update_reward_max(self):
        # self.reward_max = - \
        #     np.sum(np.bitwise_xor(self.initial_state, self.goal))+1
        # if(np.array_equal(self.goal, self.initial_state)):
        #     self.reward_max = 0

        # self.reward_max = 0 - np.sum(self.initial_state)
        self.reward_max = 0 
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
         accepts action and returns obs,reward, b_flag(episode start), info dict(optional)
        '''
        if(self.action_space.contains(action)):
            if(not(action == self.n*self.m)):
                self.state = self.bitflip(action)  # computes s_t1
            reward = self.calculate_reward(self.goal, self.state)
            self.envstepcount += 1
            done = self.compute_done(reward)
            return (np.array(self.state), reward, done, {})
        else:
            print("Invalid action")

    def bitflip(self, index):
        s2 = np.array(self.state)
        s2[index] = not s2[index]
        return s2

    def calculate_reward(self, goal, state):
        # if(np.array_equal(goal, state)):
        #     return 0.0
        # else:
        #     return -1.0
        # Below this will never execute if above isnt commented

        # For vector of all ones:
        # return -np.sum(state)*1.0

        # For minimizing max absolute auto-correlation:
        state_pm1 = state*2 - 1
        #print('State:',state_pm1)
        state_pm1 = np.reshape(state_pm1, [self.m, self.n])
        # print('State:', state_pm1)
        auto_corr_vectors = np.flip( np.fft.ifft(np.fft.fft(np.flip(state_pm1, axis=1)) * np.fft.fft(state_pm1)), axis=1 )
        # auto_corr_vectors = periodic_corr(state_pm1, state_pm1)
        # print('Auto Corr:', auto_corr_vectors)
        # max_side_peak_auto = np.max( np.abs(auto_corr_vectors[:,1:]) )
        mean_side_peak_auto = np.mean( np.square(np.abs(auto_corr_vectors[:,1:])) )
        # print('Max side:', max_side_peak_auto)
        # print()
        #input()

        # For minimizing max absolute cross-correlation:
        cross_corr_vectors = np.flip( np.fft.ifft(np.fft.fft(np.flip(state_pm1[self.first_i,:], axis=1)) * np.fft.fft(state_pm1[self.second_i,:])), axis=1 )
        # cross_corr_vectors = periodic_corr(state_pm1[self.first_i,:], state_pm1[self.second_i,:])
        # max_side_peak_cross = np.max( np.abs(cross_corr_vectors) )
        mean_side_peak_cross = np.mean( np.square(np.abs(cross_corr_vectors)) )

        # Get maximum of both
        # max_side_peak = np.maximum(max_side_peak_auto, max_side_peak_cross)
        mean_side_peak = 0.5*mean_side_peak_auto + 0.5*mean_side_peak_cross

        return -1*mean_side_peak

    def compute_done(self, reward):
        if(reward == 0 or self.envstepcount >= self.n*self.m):
            return True
        else:
            return False

    def close(self):
        pass

    def reset(self, seed=None):
        if seed == None:
            seed = self.space_seed
        self.envstepcount = 0
        # space.seed(seed)
        # self.initial_state = np.ones((self.n*self.m))
        self.initial_state = self.observation_space.sample()
        # self.goal = self.observation_space.sample()
        self.state = self.initial_state
        self.update_reward_max()
        return self.state

    def render(self, mode='human', close=False):
        print_str = str("State: "+ repr(self.state.T) +
                        " Steps done: "+str(self.envstepcount))
        if(mode == 'human'):
            print(print_str)
        if(mode == 'ansi'):
            return print_str
