import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import random

register(
    id="MultiAgentFrozenLake",
    entry_point='multi_agent_frozen_lake:MultiAgentFrozenLakeEnv',
)
def print_action(actions):
    for i, agent in enumerate(actions):
        if actions[i] == 0:
            print("Move left")
        elif actions[i] == 1:
            print("Move down")
        elif actions[i] == 2:
            print("Move right")
        elif actions[i] == 3:
            print("Move up")

def print_optimal_policy(optimal_policy, env):
    """
    Print the optimal policy grid

    Args:
    optimal_policy: The optimal policy as an array of integers
    env: The environment

    Returns:
    None
    """
    print("Optimal policy grid:")
    for i in range(0, len(optimal_policy), 8):
        # print arrows for the optimal policy
        # print the H for the hole and G for the goal in place of the arrow
        for j in range(i, i+8):
            if env.unwrapped.desc.flat[j] == b'H':
                print("H", end=" ")
            elif env.unwrapped.desc.flat[j] == b'G':
                print("G", end=" ")
            else:
                if optimal_policy[j] == 0:
                    print("←", end=" ")
                elif optimal_policy[j] == 1:
                    print("↓", end=" ")
                elif optimal_policy[j] == 2:
                    print("→", end=" ")
                elif optimal_policy[j] == 3:
                    print("↑", end=" ")
        print()
def extract_optimal_policy(q_sa):
    """Extracts the optimal policy from a Q-table.

    Args:
        q_table (numpy.ndarray): The Q-table from which to extract the policy.

    Returns:
        numpy.ndarray: The optimal policy, where policy[state] = best_action.
    """
    # Use np.argmax to find the index of the maximum Q-value for each state.
    # This index corresponds to the optimal action for that state.
    optimal_policy = np.argmax(q_sa, axis=1)
    return optimal_policy


def agent_training(env, q_sa, gamma, alpha, epsilon, episodes, num_agents):
    """
    Train the agent using Q-learning

    Args:
    env: The environment
    q_sa: The Q-table
    gamma: The discount factor
    alpha: The learning rate
    epsilon: The exploration rate
    num_episodes: The number of episodes
    num_agents: The number of agents

    Returns:
    The optimal policy
    """

    for episode in range(episodes):
        state = env.reset()
        state = state[0]
        terminated = False
        #max_q_change = 0
        while not terminated:
            actions = ()
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
                actions = action
            else:
                for agent, q in enumerate(q_sa):
                    agent_q = q[state[agent]]
                    action = np.argmax(agent_q)
                    actions += (action,)
            
            next_state, reward, terminated, truncated, info = env.step(actions)
            rewards = info['rewards']
            for agent, q in enumerate(q_sa):
                #prev_state = q[state[agent], actions[agent]]
                q[state[agent], actions[agent]] = q[state[agent], actions[agent]] + alpha * (rewards[agent] + gamma * np.max(q_sa[agent][next_state[agent]]) - q[state[agent], actions[agent]])
                #max_q_change = max(max_q_change, abs(q[state[agent], actions[agent]] - prev_state))

            state = next_state

    optimal_policy = []
    for agent in range(num_agents):
        optimal_policy.append(extract_optimal_policy(q_sa[agent]))
    return optimal_policy
     

def multi_agent_q_learning(agents, episodes, gamma, alpha, epsilon, render_mode, slippery, map_name, desc):
    # Ensure your custom environment file is in the Python path or working directory
    env = gym.make("MultiAgentFrozenLake", is_slippery=slippery, map_name=map_name, agent_positions=agents, desc=desc)

    num_agents = len(agents)

    # Initialize the environment
    env.reset()

    q_sa = [np.random.rand(env.observation_space.n, env.action_space.n) for i in range(num_agents)]

    # find terminal states
    terminal_states = []
    for state in range(env.observation_space.n):
        if env.unwrapped.desc.flat[state] in b'GH':
            terminal_states.append(state)
    
    for agent in range(num_agents):
        for state in terminal_states:
            q_sa[agent][state, :] = 0


    optimal_policy = agent_training(env, q_sa, gamma, alpha, epsilon, episodes, num_agents)
    
    win = False
    env = gym.make("MultiAgentFrozenLake", render_mode=render_mode, is_slippery=slippery, map_name=map_name, agent_positions=agents, desc=desc)
    observation, info = env.reset()
    timestep = 0

    env.render()
    while not win:
    
        actions = ()
        for agent, pos in enumerate(observation):
            actions += (optimal_policy[agent][pos],)

        observation, reward, terminated, truncated, info = env.step(actions)
        env.render()
        if reward:
            print("Reward: ", reward)
            win = True

        # Check if in a hole
        for pos in observation:
            if env.unwrapped.desc.flat[pos] == b'H':
                print("Fell in a hole")
        
        if terminated:
            print("Episode finished after {} timesteps".format(timestep+1))
            observation, info = env.reset()

        timestep += 1

    env.close()

    return 