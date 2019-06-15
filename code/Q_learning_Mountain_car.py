import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

#print(env.observation_space.high)
#print(env.observation_space.low)
#print(env.action_space.n)

learning_rate = 0.01
discount_factor = 0.95
episodes = 25000

Discrete_OS_size = [20]*len(env.observation_space.high)
discount_os_win_size = (env.observation_space.high-env.observation_space.low)/Discrete_OS_size

q_table = np.random.uniform(low = -2, high = 0, size = (Discrete_OS_size+[env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discount_os_win_size
    return tuple(discrete_state.astype(np.int))

for i in range(episodes):
    print(i)
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if i%50 is 0:    
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            
            new_q = (1- learning_rate)*current_q + learning_rate*(reward+discount_factor*max_future_q)
            q_table[discrete_state+(action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state+(action,)] = 0
        discrete_state = new_discrete_state

env.close()