import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def calculate_moving_average(data, window_size):
    """Calculate the moving average given a window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def q_experiment(env, episodes=5000, gamma=0.90, alpha=0.08, epsilon=0.1, file_name='q_exp/q_exp'):
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
            
            next_state, reward, terminated, truncated, info = env.step(action)
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
    

    
    plt.plot(episodes_grouped, cumulative_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Average Cumulative Reward per 100 Episodes')
    # Save the plot
    plt.savefig(f'{file_name}_cumulative_reward.png')
    plt.close()

    
    plt.plot(episodes_grouped, average_max_q_changes)
    plt.xlabel('Episodes')
    plt.ylabel('Average Max Q-Value Change')
    plt.title('Average Max Q-Value Change per 100 Episodes')
    # Save the plot
    plt.savefig(f'{file_name}_max_q_value_change.png')
    plt.close()
    
    plt.plot(policy_changes_smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Average Policy Changes')
    plt.title('Average Policy Stability Change per 100 Episodes')
    # Save the plot
    plt.savefig(f'{file_name}_policy_changes.png')
    plt.close()

    
    return optimal_policy, q_sa, cumulative_rewards, average_max_q_changes, policy_changes_smoothed, episodes_grouped

def policy_eval(env, experiments):
    """
    Evaluate all the experiments policies and return the best policy, breaking ties by choosing the policy with the highest average cumulative reward

    Args:
    env: Frozen Lake environment
    experiments: List of experiments to evaluate

    Returns:
    best_policy: The best policy
    """
    best_policy = None
    best_steps = float('inf')
    best_sum = float('inf')
    best_convergence = float('inf')
    for episode, g, a, e, policy, q_sa, cumulative_rewards, average_max_q_changes, policy_changes_smoothed, episodes_grouped in tqdm(experiments, desc='Evaluating Policies'):
        steps = 0
        state = env.reset()
        state = state[0]
        terminated = False
        reward = 0
        while not reward == 1:
            state, reward, terminated, truncated, info = env.step(policy[state])
            steps += 1
            if steps > 1000:
                break
        if steps < best_steps:
            best_policy = policy
            best_steps = steps
            best_sum = np.sum(cumulative_rewards)
            best_convergence = np.mean(policy_changes_smoothed)
        elif steps == best_steps:
            sum_values = np.sum(cumulative_rewards)
            if sum_values < best_sum:
                best_policy = policy
                best_steps = steps
                best_sum = sum_values
                best_convergence = np.mean(policy_changes_smoothed)
            elif sum_values == best_sum:
                if np.mean(policy_changes_smoothed) < best_convergence:
                    best_policy = policy
                    best_steps = steps
                    best_sum = sum_values
                    best_convergence = np.mean(policy_changes_smoothed)
    return best_policy

def plot_experiment(experiments, file_name='experiment'):
    """
    Plot the experiments results
    """
    for episode, g, a, e, policy, q_sa, cumulative_rewards, average_max_q_changes, policy_changes_smoothed, episodes_grouped in experiments:
        plt.plot(episodes_grouped, cumulative_rewards, label=f'{episode}_{g}_{a}_{e}')
    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Average Cumulative Reward per 100 Episodes')
    plt.legend()
    # Save the plot
    plt.savefig(f'{file_name}_cumulative_reward.png')
    plt.close()

    for episode, g, a, e, policy, q_sa, cumulative_rewards, average_max_q_changes, policy_changes_smoothed, episodes_grouped in experiments:
        plt.plot(episodes_grouped, average_max_q_changes, label=f'{episode}_{g}_{a}_{e}')
    plt.xlabel('Episodes')
    plt.ylabel('Average Max Q-Value Change')
    plt.title('Average Max Q-Value Change per 100 Episodes')
    plt.legend()
    # Save the plot
    plt.savefig(f'{file_name}_max_q_value_change.png')
    plt.close()

    for episode, g, a, e, policy, q_sa, cumulative_rewards, average_max_q_changes, policy_changes_smoothed, episodes_grouped in experiments:
        plt.plot(policy_changes_smoothed, label=f'{episode}_{g}_{a}_{e}')
    plt.xlabel('Episode')
    plt.ylabel('Average Policy Changes')
    plt.title('Average Policy Stability Change per 100 Episodes')
    plt.legend()
    # Save the plot
    plt.savefig(f'{file_name}_policy_changes.png')
    plt.close()


def q_learning(env, episodes=5000, gamma=0.90, alpha=0.08, epsilon=0.1, experiment=False):
    if experiment:
        print("Running experiment")
        episodes = [100, 1000, 5000, 10000]
        gamma = [0.7, 0.90, 0.99]
        alpha = [0.01, 0.3, 0.9]
        epsilon = [0.1, 0.3, 0.5]
    else:
        episodes = [episodes]
        gamma = [gamma]
        alpha = [alpha]
        epsilon = [epsilon]
    experiments = []

    for episode in episodes:
        for g in gamma:
            for a in alpha:
                for e in epsilon:
                    file_name = f'q_exp/{episode}_{g}_{a}_{e}'
                    optimal_policy, q_sa, cumulative_rewards, average_max_q_changes, policy_changes_smoothed, episodes_grouped = q_experiment(env, episode, g, a, e, file_name)
                    experiments.append((episode, g, a, e, optimal_policy, q_sa, cumulative_rewards, average_max_q_changes, policy_changes_smoothed, episodes_grouped))
    
    if experiment:
        plot_experiment(experiments, file_name='q_exp/experiment')


    best_policy = policy_eval(env, experiments)
    return best_policy



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