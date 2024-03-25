# import numpy as np
# from tqdm import tqdm
# import random

# def evalmcev(env, episodes=100000, gamma=0.9, epsilon=0.2):
#     """
#     Monte Carlo algorithm to solve the Frozen Lake environment
#     I'm using code from https://gist.github.com/yunsangq/b3e4c4b51c6c8b0dbae621b6b92c5cb5
#     to compare my approach with the author's approach.
#     To be clearn, I had to modify the code to make it work with my Frozen Lake environment.
#     """
#     Q = np.random.rand(env.observation_space.n, env.action_space.n)  # Random initialization
#     for state in range(env.observation_space.n):
#         if env.unwrapped.desc.flat[state] in b'GH':  # Terminal states adjustment
#             Q[state] = np.zeros(env.action_space.n)
    
#     n_s_a = np.zeros([env.observation_space.n, env.action_space.n])
#     successes = 0

#     for _ in tqdm(range(episodes)):
#         state = env.reset()
#         state = state[0]
#         episode = []
#         done = False
#         while not done:
#             action = np.random.choice([env.action_space.sample(), np.argmax(Q[state])], p=[epsilon, 1-epsilon])
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             episode.append((state, action, reward))
#             state = next_state
#             done = terminated or truncated
#             if done and reward == 1:  # Assuming a reward of 1 indicates success
#                 successes += 1

#         # Update Q-values based on the episode
#         G = 0
#         for state, action, reward in reversed(episode):
#             G = gamma * G + reward
#             Q[state, action] = Q[state, action] + (1.0 / (1 + n_s_a[state, action])) * (G - Q[state, action])
#             n_s_a[state, action] += 1

#     # Policy derivation from Q-values
#     policy = np.argmax(Q, axis=1)

#     print("Success rate: ", successes / episodes)

#     return policy

import gymnasium as gym
from gymnasium import wrappers
import numpy as np

# Monte Carlo algorithm to solve the Frozen Lake environment
# I'm using code from https://gist.github.com/yunsangq/b3e4c4b51c6c8b0dbae621b6b92c5cb5
# to compare my approach with the author's approach.
# To be clearn, I had to modify the code to make it work with my Frozen Lake environment.

env = gym.make("FrozenLake-v1")

Q = np.zeros([env.observation_space.n, env.action_space.n])
# Find terminal states
terminal_states = []
for state in range(env.observation_space.n):
    if env.unwrapped.desc.flat[state] in b'GH':
        terminal_states.append(state)

# Assign 0 to terminal states in Q(s, a)
for state in terminal_states:
    Q[state] = np.zeros(env.action_space.n)

n_s_a = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 5000
epsilon = 0.1
rList = []

for i in range(num_episodes):
    state = env.reset()
    state = state[0]
    rAll = 0
    done = False
    results_list = []
    result_sum = 0.0
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        results_list.append((state, action))
        result_sum += reward
        state = next_state
        rAll += reward
        done = terminated or truncated
    rList.append(rAll)

    for (state, action) in results_list:
        n_s_a[state, action] += 1.0
        alpha = 1.0 / n_s_a[state, action]
        Q[state, action] += alpha * (result_sum - Q[state, action])

print("Final success rate: " + str(sum(rList) / num_episodes))

# Print optimal policy
print("Optimal policy:")
print(np.argmax(Q, axis=1))

env.close()