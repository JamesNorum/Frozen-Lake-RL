import numpy as np
from tqdm import tqdm
import random


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

    for _ in tqdm(range(episodes)):
        state = env.reset()
        state = state[0]
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_sa[state])
            
            next_state, reward, done, truncated, info  = env.step(action)
            q_sa[state, action] = q_sa[state, action] + alpha * (reward + gamma * np.max(q_sa[next_state]) - q_sa[state, action])
            state = next_state
            if done:
                break

    optimal_policy = np.argmax(q_sa, axis=1)
        
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