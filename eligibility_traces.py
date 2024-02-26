import tqdm as tqdm
import numpy as np
from monte_carlo import init_q_sa

# Choose an action using an Epsilon-Greedy policy
def choose_action(env, Q, epsilon, state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])
    
# Choose optimal policy
def choose_optimal_policy(Q):
    policy = np.argmax(Q, axis=1)
    return policy  

def sarsa_lambda_et(env, episodes=5000, gamma=0.90, alpha=0.08, epsilon=0.1, lambda_=0.9):
    # Initialize the environment Frozen Lake V1
    env = env

    # Parameters
    episodes = episodes  # Number of episodes to run
    gamma = gamma  # Discount factor
    alpha = alpha  # Learning rate
    epsilon = epsilon  # Exploration rate
    lambda_ = lambda_  # Eligibility trace decay rate

    # Initialize Q-table and Eligibility Traces
    Q = init_q_sa(env)
    E = np.zeros((env.observation_space.n, env.action_space.n))

    for _ in tqdm.tqdm(range(episodes)):
        # Initialize S to the initail state of the environment
        state = env.reset()
        state = state[0] 
        # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
        action = choose_action(env, Q, epsilon, state)
        
        # Set flag for termination
        terminated = False
        while not terminated:
            # Take action A, observe R, S'
            next_state, reward, terminated, truncated, info = env.step(action)
            # Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
            next_action = choose_action(env, Q, epsilon, next_state)
            # Calculate delta (TD error)
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            # Increment eligibility trace
            E[state, action] += 1

            # Update Q and E for all states and actions
            for s in range(env.observation_space.n):
                for a in range(env.action_space.n):
                    # Update Q-value using the step size (α), the difference (δ), and the eligibility trace.
                    Q[s, a] += alpha * delta * E[s, a]
                    # Decay the eligibility trace, controlled by the discount factor (γ) and the trace decay rate (λ).
                    E[s, a] *= gamma * lambda_
            
            # Move to next state and action
            state, action = next_state, next_action

        # Choose optimal policy
        policy = choose_optimal_policy(Q)

    # Print the optimal policy
    return policy