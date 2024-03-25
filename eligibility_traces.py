import tqdm as tqdm
import numpy as np
from monte_carlo import init_q_sa
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    """ Compute moving average """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

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

    # Variables used to track metrics
    total_rewards = []
    successes = []
    cumulative_discounted_rewards = []

    for _ in tqdm.tqdm(range(episodes)):
        # Initialize S to the initail state of the environment
        state = env.reset()
        state = state[0] 
        # Choose A from S using policy derived from Q (e.g., epsilon-greedy)
        action = choose_action(env, Q, epsilon, state)
    
        # Reset metrics at the start of each episode
        total_reward = 0
        cumulative_discounted_reward = 0  # Reset at the start of each episode
        gamma_t = 1  # Discount factor power index (timestep)

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
            
            # Update metrics
            total_reward += reward
            cumulative_discounted_reward += gamma_t * reward
            gamma_t *= gamma

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

        # Track total reward and whether the episode was successful
        total_rewards.append(total_reward)
        successes.append(1 if reward > 0 else 0)  # Assuming positive reward indicates success
        cumulative_discounted_rewards.append(cumulative_discounted_reward)

    # Calculate metrics
    average_rewards = [np.mean(total_rewards[max(0, i-100):i+1]) for i in range(len(total_rewards))]
    success_rate = moving_average(successes, 100)  # 100-episode window for success rate
    window_size = episodes // 10
    smoothed_average_rewards = moving_average(average_rewards, window_size)
    smoothed_cumulative_discounted_rewards = moving_average(cumulative_discounted_rewards, window_size)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

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

    # Adjust layout to prevent overlapping
    plt.tight_layout(pad=4.0)  # Adjust padding

    plt.savefig('eligibility_traces.png')  # Saving the figure
    
    # Return the optimal policy
    return policy