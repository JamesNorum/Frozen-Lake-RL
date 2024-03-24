"""
This code was adapted from:
https://github.com/johnnycode8/gym_solutions/blob/main/frozen_lake_q.py
The only modifications made were to pass in the environment and episodes from the main function to make sure it is the same environment.
Also it now returns the optimal policy
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(env, episodes, is_training=True, render=False):

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    cumulative_rewards = []
    average_max_q_changes = []
    rewards_temp = []
    max_q_changes_temp = []
    policy_changes_smoothed = []
    policy_changes_temp = []
    previous_policy = np.argmax(q, axis=1)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200
        total_reward = 0
        max_q_change = 0

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)
            
            prev_state = q[state, action]

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state
            total_reward += reward
            max_q_change = max(max_q_change, abs(q[state, action] - prev_state))

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

        current_policy = np.argmax(q, axis=1)
        policy_change_count = np.sum(current_policy != previous_policy)
        previous_policy = current_policy
        policy_changes_temp.append(policy_change_count)
        rewards_temp.append(total_reward)
        max_q_changes_temp.append(max_q_change)

        if (i + 1) % 100 == 0:
            cumulative_rewards.append(np.mean(rewards_temp))
            average_max_q_changes.append(np.mean(max_q_changes_temp))
            policy_changes_smoothed.append(np.mean(policy_changes_temp))
            rewards_temp = []
            max_q_changes_temp = []
            policy_changes_temp = []

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

    episodes_grouped = range(0, episodes, 100)
    
    plt.plot(episodes_grouped, cumulative_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Average Cumulative Reward per 100 Episodes')
    # Save the plot
    plt.savefig('eval_cumulative_reward.png')
    plt.close()

    
    plt.plot(episodes_grouped, average_max_q_changes)
    plt.xlabel('Episodes')
    plt.ylabel('Average Max Q-Value Change')
    plt.title('Average Max Q-Value Change per 100 Episodes')
    # Save the plot
    plt.savefig('eval_max_q_value_change.png')
    plt.close()
    
    plt.plot(policy_changes_smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Average Policy Changes')
    plt.title('Average Policy Stability Change per 100 Episodes')
    # Save the plot
    plt.savefig('eval_policy_changes.png')
    plt.close()

    optimal_policy = np.argmax(q, axis=1)
    return optimal_policy

def eval_q(env, episode):
    optimal_policy = run(env, episode)
    return optimal_policy



if __name__ == '__main__':
    # run(15000)

    run(1000, is_training=True, render=True)