import gymnasium as gym
import numpy as np
from numpy.random import random, choice
import matplotlib.pyplot as plt

def epsilon_greedy(state, Q, epsilon):

  values = Q[state,:]
  max_value = max(values)
  no_actions = len(values)

  greedy_actions = [a for a in range(no_actions) if values[a] == max_value]

  explore = (random() < epsilon)

  if explore:
    return choice([a for a in range(no_actions)])
  else:
    return choice([a for a in greedy_actions])

def moving_average(data, window_size):
    """ Compute moving average """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def eval_et(env, episodes=5000, gamma=0.90, alpha=0.08, epsilon=0.1, lambda_=0.9):
  """
  SARSA lambda algorithm to solve the Frozen Lake environment
  I'm using code from https://github.com/adieu2/sarsa-lambda-frozen-lake/tree/master
  to compare my approach with the author's approach.
  To be clearn, I had to modify the code to make it work with my Frozen Lake environment
  as well as track the same metrics I tracked in my implementation.
  """
  # parameters for sarsa(lambda)
  env = env
  episodes = episodes
  gamma = gamma
  alpha = alpha
  epsilon = epsilon
  epsilon_decay = 0.999 # decay per episode
  eligibility_decay = lambda_

  no_states = env.observation_space.n
  no_actions = env.action_space.n
  Q = np.zeros((no_states, no_actions))

  # New metrics
  total_rewards = []
  successes = []
  cumulative_discounted_rewards = []

  for episode in range(episodes):

    state = env.reset()
    state = state[0]
    epsilon *= epsilon_decay
    #epsilon = epsilon

    action = epsilon_greedy(state, Q, epsilon)

    R = [None] # No first return
    E = np.zeros((no_states, no_actions))

    total_reward = 0
    cumulative_discounted_reward = 0
    gamma_t = 1  # Discount factor power index (timestep)
    
    while True:

      E = eligibility_decay * gamma * E
      E[state, action] += 1

      new_state, reward, terminated, truncated, info = env.step(action)
      new_action = epsilon_greedy(new_state, Q, epsilon)

      R.append(reward)

      delta = reward + gamma * Q[new_state, new_action] - Q[state, action]
      Q = Q + alpha * delta * E 

      state, action = new_state, new_action

      total_reward += reward
      cumulative_discounted_reward += gamma_t * reward
      gamma_t *= gamma

      if terminated or truncated:
        break
      
    total_rewards.append(total_reward)
    successes.append(1 if reward > 0 else 0)
    cumulative_discounted_rewards.append(cumulative_discounted_reward)

  # Calculate metrics for plots
  average_rewards = [np.mean(total_rewards[max(0, i-100):i+1]) for i in range(len(total_rewards))]
  success_rate = moving_average(successes, 100)
  window_size = 100  # Smoothing window size
  smoothed_average_rewards = moving_average(average_rewards, window_size)
  smoothed_cumulative_discounted_rewards = moving_average(cumulative_discounted_rewards, window_size)

  # Plotting
  fig, axs = plt.subplots(3, 1, figsize=(10, 15))

  # Average Reward per Episode
  axs[0].plot(smoothed_average_rewards, label='Average Reward per Episode', color='orange')
  axs[0].set_ylabel('Average Reward')
  axs[0].set_title('Average Reward per Episode over Training')
  axs[0].legend()
  axs[0].grid(True)

  # Success Rate Over Time
  axs[1].plot(success_rate, label='Success Rate', color='blue')
  axs[1].set_ylabel('Success Rate')
  axs[1].set_title('Success Rate Over Time')
  axs[1].legend()
  axs[1].grid(True)

  # Cumulative Discounted Reward
  axs[2].plot(smoothed_cumulative_discounted_rewards, label='Cumulative Discounted Reward', color='green')
  axs[2].set_xlabel('Episode')
  axs[2].set_ylabel('Cumulative Discounted Reward')
  axs[2].set_title('Cumulative Discounted Reward per Episode')
  axs[2].legend()
  axs[2].grid(True)

  plt.tight_layout(pad=4.0)
  plt.savefig('eligibility_traces_eval.png')  # Saving the figure

  # Return Optimal Policy
  return(np.argmax(Q, axis=1))