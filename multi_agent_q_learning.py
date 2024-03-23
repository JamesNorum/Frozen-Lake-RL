import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

register(
    id="MultiAgentFrozenLake",
    entry_point='multi_agent_frozen_lake:MultiAgentFrozenLakeEnv',
)

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

def plot_win_rate(win_rate, num_agents):
    """
    Plot the win rate

    Args:
    win_rate: The win rate

    Returns:
    None
    """
    for agent, wins in win_rate.items():
        plt.plot(wins, label='Agent ' + str(agent + 1))
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.title('Win Rate vs Episodes')
    plt.legend()
    # Save the plot
    plt.savefig(f'win_rate_{num_agents}.png')

def plot_average_reward(average_reward, num_agents):
    """
    Plot the average reward

    Args:
    average_reward: The average reward

    Returns:
    None
    """
    for agent, rewards in average_reward.items():
        plt.plot(rewards, label='Agent ' + str(agent + 1))
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Episodes')
    plt.legend()
    # Save the plot
    plt.savefig(f'average_reward_{num_agents}.png')

def plot_convergence_time(convergence_time, q_changes, num_agents):
    """
    Plot the convergence time

    Args:
    convergence_time: The convergence time

    Returns:
    None
    """
    for agent, time in q_changes.items():
        plt.plot(time, label='Agent ' + str(agent + 1))
    print(q_changes)
    # Plot the convergence time points, it is a single point at episode x, so agent 1 might converge at episode 10, agent 2 at episode 20, etc.
    #for agent, time in convergence_time.items():
        #rint(q_changes[agent][time])
        #plt.scatter(time, q_changes[agent][time], label='Agent ' + str(agent + 1) + ' Convergence Time', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Q-Value Change')
    plt.title('Q-Value Change vs Episodes')
    plt.legend()
    # Save the plot
    plt.savefig(f'q_changes_{num_agents}.png')

def agent_training(env, q_sa, gamma, alpha, epsilon, episodes, num_agents, block=False, blocking_penalty=-0.1):
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

    win_rate = {i: [] for i in range(num_agents)}
    average_reward = {i: [] for i in range(num_agents)}
    convergence_time = {i: 0 for i in range(num_agents)}
    max_q_changes = {i: [] for i in range(num_agents)}
    for episode in tqdm(range(episodes)):
        state = env.reset()
        state = state[0]
        terminated = False
        win_count = [0 for i in range(num_agents)]
        agent_average_reward = {i: 0 for i in range(num_agents)}
        max_q_change = [0 for i in range(num_agents)]
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
            
            for agent, r in enumerate(rewards):
                if info['goal_reached'][agent]:
                    win_count[agent] += 1

            for agent, q in enumerate(q_sa):
                agent_average_reward[agent] += rewards[agent]
                if block:
                    if agent in info.get("blocking", {}):
                        rewards[agent] += blocking_penalty
                prev_state = q[state[agent], actions[agent]]
                q[state[agent], actions[agent]] = q[state[agent], actions[agent]] + alpha * (rewards[agent] + gamma * np.max(q_sa[agent][next_state[agent]]) - q[state[agent], actions[agent]])
                max_q_change[agent] = max(max_q_change[agent], abs(prev_state - q[state[agent], actions[agent]]))
                
            state = next_state

            for agent in range(num_agents):
               if max_q_change[agent] < 1e-4 and convergence_time[agent] == 0:
                    print("Agent ", agent + 1, " converged at episode ", episode)
                    convergence_time[agent] = episode

        for agent in range(num_agents):
            average_reward[agent].append(agent_average_reward[agent])
            max_q_changes[agent].append(max_q_change)

        for agent, wins in enumerate(win_count):
            win_rate[agent].append(wins / (episode + 1))

    optimal_policy = []
    for agent in range(num_agents):
        optimal_policy.append(np.argmax(q_sa[agent], axis=1))
    return optimal_policy, win_rate, average_reward, convergence_time, max_q_changes
     

def multi_agent_q_learning(agents, episodes, gamma, alpha, epsilon, render_mode, slippery, map_name, desc, block=False, blocking_penalty=-0.1, goal_reward=1, hole_reward=0, step_reward=0):
    # Ensure your custom environment file is in the Python path or working directory
    env = gym.make("MultiAgentFrozenLake", is_slippery=slippery, map_name=map_name, agent_positions=agents, desc=desc, goal_reward=goal_reward, hole_reward=hole_reward, step_reward=step_reward)

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


    optimal_policy, win_rate, average_reward, convergence_time, max_q_changes = agent_training(env, q_sa, gamma, alpha, epsilon, episodes, num_agents, block, blocking_penalty)

    plot_win_rate(win_rate, num_agents)
    plot_average_reward(average_reward, num_agents)
    plot_convergence_time(convergence_time, max_q_changes, num_agents)

    for agent in range(num_agents):
        print("Agent ", agent + 1, " optimal policy:")
        print_optimal_policy(optimal_policy[agent], env)
    
    win = False
    env = gym.make("MultiAgentFrozenLake", render_mode=render_mode, is_slippery=slippery, map_name=map_name, agent_positions=agents, desc=desc, goal_reward=goal_reward, hole_reward=hole_reward, step_reward=step_reward)
    observation, info = env.reset()
    timestep = 0

    env.render()
    agent_termination = [False for i in range(num_agents)]
    while not win:
    
        actions = ()
        for agent, pos in enumerate(observation):
            actions += (optimal_policy[agent][pos],)

        observation, reward, terminated, truncated, info = env.step(actions)

        rewards = info['rewards']

        env.render()

        for agent, r in enumerate(rewards):
            if info['goal_reached'][agent]:
                print("Reward: ", r)
                print("Won after {} timesteps".format(timestep+1))
                print("Agent ", agent + 1, " wins")
                win = True

        # Check if in a hole
        for pos in observation:
            if env.unwrapped.desc.flat[pos] == b'H':
                #print("Fell in a hole")
                pass
        agent_terminated = info['terminated']

        for agent, term in enumerate(agent_terminated):
            if agent_terminated[agent]:
                agent_termination[agent] = True

        if all(agent_termination):
            print("Episode finished after {} timesteps".format(timestep+1))
            agent_termination = [False for i in range(num_agents)]
            observation, info = env.reset()

        timestep += 1

    env.close()

    return 