import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict

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

def monte_carlo(env, episodes=5000, gamma=0.9, epsilon=0.1):
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
        
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            returns_sum[state][action] += G
            returns_count[state][action] += 1
            Q_sa[state][action] = returns_sum[state][action] / returns_count[state][action]
        
        policy = generate_policy(Q_sa)

    return policy