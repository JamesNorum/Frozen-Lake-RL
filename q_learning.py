import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def calculate_moving_average(data, window_size):
    """Calculate the moving average given a window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def q_learning(env, episodes=5000, gamma=0.90, alpha=0.08, epsilon=0.1):
    """
    Q-Learning algorithm to solve the Frozen Lake environment

    Args:
    env: OpenAI Gym environment
    episodes: Number of episodes
    gamma: Discount factor
    alpha: Alpha Value
    epsilon: Epsilon Value

    Returns:
    optimal_policy: The optimal policy as an array of integers
    """

    # find terminal states
    terminal_states = []
    for state in range(env.observation_space.n):
        if env.unwrapped.desc.flat[state] in b'GH':
            terminal_states.append(state)

    # Initialize Q(s, a) with 0 to 3 for all but terminal states, terminal states are 0
    q_sa = np.random.rand(env.observation_space.n, env.action_space.n)
    
    for state in terminal_states:
        q_sa[state] = np.zeros(env.action_space.n)
    
    cumulative_rewards = []
    average_max_q_changes = []
    rewards_temp = []
    max_q_changes_temp = []
    policy_changes_smoothed = []
    policy_changes_temp = []
    previous_policy = np.argmax(q_sa, axis=1)

    for episode in tqdm(range(episodes)):
        state = env.reset()
        state = state[0]
        terminated = False
        total_reward = 0
        max_q_change = 0
        

        while not terminated:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_sa[state])
            
            next_state, reward, terminated, truncated, info  = env.step(action)
            prev_state = q_sa[state, action]
            q_sa[state, action] = q_sa[state, action] + alpha * (reward + gamma * np.max(q_sa[next_state]) - q_sa[state, action])
            max_q_change = max(max_q_change, abs(q_sa[state, action] - prev_state))

            state = next_state
            total_reward += reward
            
        
        current_policy = np.argmax(q_sa, axis=1)
        policy_change_count = np.sum(current_policy != previous_policy)
        previous_policy = current_policy
        policy_changes_temp.append(policy_change_count)
        rewards_temp.append(total_reward)
        max_q_changes_temp.append(max_q_change)

        if (episode + 1) % 100 == 0:
            cumulative_rewards.append(np.mean(rewards_temp))
            average_max_q_changes.append(np.mean(max_q_changes_temp))
            policy_changes_smoothed.append(np.mean(policy_changes_temp))
            rewards_temp = []
            max_q_changes_temp = []
            policy_changes_temp = []

    optimal_policy = np.argmax(q_sa, axis=1)


     # Plotting
    episodes_grouped = range(0, episodes, 100)
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 2, 1)
    plt.plot(episodes_grouped, cumulative_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Average Cumulative Reward per 100 Episodes')

    plt.subplot(2, 2, 2)
    plt.plot(episodes_grouped, average_max_q_changes)
    plt.xlabel('Episodes')
    plt.ylabel('Average Max Q-Value Change')
    plt.title('Average Max Q-Value Change per 100 Episodes')

    plt.subplot(2, 2, 3)
    plt.plot(policy_changes_smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Average Policy Changes')
    plt.title('Average Policy Stability Change per 100 Episodes')


    plt.tight_layout()
    # Save the plot
    plt.savefig("q_learning.png")

        
    return optimal_policy


"""
Algorithm parameters: step size alpha in (0, 1], small epsilon > 0
Initialize Q(s, a), for all s in S+ , a in A(s), arbitrarily except that Q(terminal, •) = 0
Loop for each episode:
    Initialize S
    Loop for each step of episode:
        Choose A from S using policy derived from Q (e.g., e-greedy)
        Take action A, observe R, S'
        Q(S, A) < Q(S, A) + alpha[R + gamma * max_a Q(S', a) - Q(S, A)]
        S<-S'
    until S is terminal
"""