#!/usr/bin/env python
# coding: utf-8

# In[50]:


import gym
import numpy as np
import random


# In[51]:


# load environment
env = gym.make("FrozenLake-v0")
env.reset()
env.render()


# In[91]:


# take action and observaton size
action_size = env.action_space.n
obs_size    = env.observation_space.n
print(action_size, obs_size)


# In[157]:


# set variables
episodes = 2500
max_steps = 20
learning_rate = 0.5
discount = 0.9
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
epsilon_decayRate = (max_epsilon-min_epsilon)/episodes
q_table = np.zeros((obs_size, action_size))


# In[156]:


for i in range(episodes):
    cur_state = env.reset()
    done = False
    print("+---------------------------+")
    print(i)
    j = 0 
    while done is False and j< max_steps:
        env.render()
        # choose exploration vs exploitation
        x = np.random.rand()
        if(x<epsilon):
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[cur_state,:])

        # change epsilon    
        if epsilon <= max_epsilon and epsilon >= min_epsilon:
            epsilon = epsilon - epsilon_decayRate

        # update the state
        new_state, reward, done, info = env.step(action)

        q_table[cur_state,action] += learning_rate*(reward + discount*max(q_table[new_state,:]) - q_table[cur_state,action])

        cur_state = new_state
        j = j+1
    

