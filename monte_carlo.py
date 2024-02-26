import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    """ Compute moving average """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def generate_episode(env, policy, epsilon):
    episode = []
    state = env.reset()
    state = state[0]
    done = False
    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = policy[state]  # Exploit based on current policy
        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    return episode

def generate_policy(Q_sa):
    return np.argmax(Q_sa, axis=1)

def init_q_sa(env):
    # Initialize Q(s, a) with random values, except for in terminal states
    Q_sa = np.random.rand(env.observation_space.n, env.action_space.n)

    # Find terminal states
    terminal_states = []
    for state in range(env.observation_space.n):
        if env.unwrapped.desc.flat[state] in b'GH':
            terminal_states.append(state)

    # Assign 0 to terminal states in Q(s, a)
    for state in terminal_states:
        Q_sa[state] = np.zeros(env.action_space.n)

    return Q_sa    

def every_visit_monte_carlo(env, episodes=5000, gamma=0.9, epsilon=0.1):
    # Define Constants
    env = env
    episodes = episodes
    gamma = gamma
    epsilon = epsilon

    # Initialize Q(s,a)
    Q_sa = init_q_sa(env)

    # Initialize policy based on Q(s, a)
    policy = generate_policy(Q_sa)

    # Initialize empty dictionaries for returns 
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))

    # Initialize lists to track metrics
    episode_returns = []  # Total return (sum of rewards) per episode
    episode_lengths = []  # Number of steps per episode
    average_rewards_per_step = []  # Average reward per step in an episode

    for _ in tqdm(range(episodes)):
        episode = generate_episode(env, policy, epsilon)
        G = 0
        total_return = 0 # Total return for this episode - used for graphs
        
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            returns_sum[state][action] += G
            returns_count[state][action] += 1
            Q_sa[state][action] = returns_sum[state][action] / returns_count[state][action]
            total_return += reward  # Accumulate total return for this episode - used for graphs
        
        policy = generate_policy(Q_sa)
        
        # Track metrics for this episode - used for graphs
        episode_returns.append(total_return)
        episode_lengths.append(len(episode))
        average_rewards_per_step.append(total_return / len(episode) if len(episode) > 0 else 0)

    
    # Set window size to 1% of the number of episodes
    window_size = episodes // 10  

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))  # Adjusted for better fit

    # Plot Total Return per Episode with Moving Average
    smoothed_returns = moving_average(episode_returns, window_size)
    axs[0].plot(smoothed_returns, label='Total Return per Episode (Smoothed)')
    axs[0].set_ylabel('Total Return (Smoothed)')
    axs[0].set_title('Total Return per Episode over Training (Smoothed)')
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    # Plot Episode Length over Training with Moving Average
    smoothed_lengths = moving_average(episode_lengths, window_size)
    axs[1].plot(smoothed_lengths, label='Episode Length (Smoothed)', color='orange')
    axs[1].set_ylabel('Episode Length (Smoothed)')
    axs[1].set_title('Episode Length over Training (Smoothed)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Average Reward Per Step with Moving Average
    #plt.figure(figsize=(12, 7))
    smoothed_rewards = moving_average(average_rewards_per_step, window_size)
    axs[2].plot(smoothed_rewards, label='Average Reward Per Step (Smoothed)', color='green')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Average Reward Per Step (Smoothed)')
    axs[2].set_title('Average Reward Per Step over Training (Smoothed)')
    axs[2].legend(loc='lower right')
    axs[2].grid(True)

    # Adjust layout to prevent overlapping
    plt.tight_layout(pad=4.0)  # Adjust padding

    plt.savefig('monte_carlo_ev.png')

    return policy

def first_visit_monte_carlo(env, episodes=5000, gamma=0.9, epsilon=0.1):
    # Define Constants
    env = env
    episodes = episodes
    gamma = gamma
    epsilon = epsilon

    # Initialize Q(s,a)
    Q_sa = init_q_sa(env)

    # Initialize policy based on Q(s, a)
    policy = generate_policy(Q_sa)

    # Initialize empty dictionaries for returns 
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in tqdm(range(episodes)):
        episode = generate_episode(env, policy, epsilon)
        G = 0
        visited_state_actions = set()  # To track first visits

        for state, action, reward in reversed(episode):
            if (state, action) not in visited_state_actions:
                G = gamma * G + reward
                returns_sum[state][action] += G
                returns_count[state][action] += 1
                Q_sa[state][action] = returns_sum[state][action] / returns_count[state][action]
                visited_state_actions.add((state, action))  # Mark this state-action as visited
        
        policy = generate_policy(Q_sa)

    return policy