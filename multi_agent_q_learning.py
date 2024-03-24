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

def plot_win_count(win_count, num_agents):
    """
    Plot a histogram of the win count, agents on the x-axis and win count on the y-axis

    Args:
    win_count: The win count
    num_agents: The number of agents

    Returns:
    None
    """
    # Plot the agents on the x-axis and the win count on the y-axis, for each agent place their bars at their agent number on the x-axis
    for agent, wins in win_count.items():
        plt.bar(agent, len(wins), label='Agent ' + str(agent + 1))
    
    plt.xlabel('Agent')
    plt.ylabel('Count')
    plt.title('Win Count')
    plt.legend()
    # Save the plot
    plt.savefig(f'win_count_{num_agents}.png')
    plt.close()
    
def reward_per_agent(reward, num_agents):
    """
    Plot the average reward

    Args:
    average_reward: The average reward

    Returns:
    None
    """
    average_reward = {i: [] for i in range(num_agents)}
    # average the rewards for each agent over every 100 episodes
    for agent, rewards in reward.items():
        average_reward[agent] = [sum(rewards[i:i+100])/100 for i in range(0, len(rewards), 100)]
        
    for agent, rewards in average_reward.items():
        plt.plot(rewards, label='Agent ' + str(agent + 1))
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Agent Reward vs Episodes')
    plt.legend()
    # Save the plot
    plt.savefig(f'average_reward_per_agent_{num_agents}.png')
    plt.close()

def plot_convergence_time(convergence_time, q_changes, num_agents):
    """
    Plot the convergence time

    Args:
    convergence_time: The convergence time

    Returns:
    None
    """
    average_q_changes = {i: [] for i in range(num_agents)}
    # average the q-value changes for each agent over every 100 episodes
    for agent, q_changes in q_changes.items():
        average_q_changes[agent] = [sum(q_changes[i:i+100])/100 for i in range(0, len(q_changes), 100)]

    # plot the average q-value changes for each agent over the enire episode count
    # time is the episodes skipped by 100
    time = [i for i in range(0, len(q_changes), 100)]
    for agent, q_changes in average_q_changes.items():
        plt.plot(time, q_changes, label='Agent ' + str(agent + 1))

    for agent, time in convergence_time.items():
        if time == 0:
            print("Agent ", agent + 1, " did not converge")
            # plot at the top left of the plot that the agent did not converge
            plt.axvline(-100, 100, color='r', linestyle='--', label='Agent ' + str(agent + 1) + ' Did Not Converge')
            continue
        print("Agent ", agent + 1, " converged at episode ", time)
        plt.axvline(x=time, color='r', linestyle='--', label='Agent ' + str(agent + 1) + ' Convergence Time')
        plt.text(time, 0.2, 'Agent ' + str(agent + 1) + 'Time: ' + str(time) , rotation=90)
        
    
    plt.xlabel('Episodes')
    plt.ylabel('Q-Value Change')
    plt.title('Q-Value Change vs Episodes')
    
    plt.legend(loc='upper right')
    # Save the plot
    plt.savefig(f'q_changes_{num_agents}.png')
    plt.close()

def plot_reward_trends(rewards, num_agents):
    """
    Plot the reward trend

    Args:
    rewards: The rewards
    num_agents: The number of agents

    Returns:
    None
    """
    average_reward = [sum(rewards[i:i+100])/100 for i in range(0, len(rewards), 100)]
    time = [i for i in range(0, len(rewards), 100)]

    plt.figure(figsize=(10, 5))
    plt.plot(time, average_reward, label='Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Trend Over Time')
    plt.legend()
    plt.savefig(f'reward_trend_{num_agents}.png')
    plt.close()



def position_to_coordinates(position, ncols):
    """Convert a linear position to 2D (row, col) coordinates."""
    return divmod(position, ncols)

def plot_trajectories(agent_positions, nrows, ncols, holes, goal, num_agents, episode=None):
    """
    Plot the trajectories of agents, including holes and the goal, with numbered grid squares and padding around the grid.

    Args:
    - agent_positions: Dictionary with keys as agent IDs and values as lists of lists of positions for each episode.
    - nrows: The number of rows in the grid.
    - ncols: The number of columns in the grid.
    - holes: A list of positions representing holes in the grid.
    - goal: The position of the goal in the grid.
    - episode: Optional integer specifying which episode to plot. If None, plots all episodes.
    """
    padding = 0.5  # Padding around the grid
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_xlim(-padding, ncols-1+padding)
    ax.set_ylim(-padding, nrows-1+padding)
    # Invert y-axis to match the grid layout
    ax.invert_yaxis()
    
    # Drawing the grid lines
    ax.set_xticks(np.arange(ncols), minor=False)
    ax.set_yticks(np.arange(nrows), minor=False)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
    # Labeling the grid squares
    ax.set_xticklabels(np.arange(ncols))
    ax.set_yticklabels(np.arange(nrows))

    # Plot holes
    for hole in holes:
        hole_row, hole_col = position_to_coordinates(hole, ncols)
        ax.scatter(hole_col, hole_row, color='black', marker='X', s=100, label='Hole' if hole == holes[0] else "")

    # Plot goal
    goal_row, goal_col = position_to_coordinates(goal, ncols)
    ax.scatter(goal_col, goal_row, color='gold', marker='*', s=200, label='Goal')

    colors = plt.cm.viridis(np.linspace(0, 1, len(agent_positions)))

    for agent_id, all_positions in agent_positions.items():
        if episode is not None:
            positions = all_positions[episode]
            rows, cols = zip(*[position_to_coordinates(pos, ncols) for pos in positions])
            ax.plot(cols, rows, '-o', label=f'Agent {agent_id+1}, Episode {episode+1}', color=colors[agent_id])
        else:
            for ep, positions in enumerate(all_positions):
                rows, cols = zip(*[position_to_coordinates(pos, ncols) for pos in positions])
                ax.plot(cols, rows, '-o', alpha=0.3 + 0.7 * (ep / len(all_positions)), color=colors[agent_id], label=f'Agent {agent_id+1}' if ep == 0 else '')

    ax.legend()
    # Save the plot
    if episode is not None:
        plt.savefig(f'trajectories_episode_{episode}_{num_agents}.png')
    else:
        plt.savefig('trajectories_all_episodes.png')
    plt.close()


 import seaborn as sns

def plot_q_values(q_table, action_names):
    fig, axs = plt.subplots(1, len(action_names), figsize=(20, 5))
    for idx, action in enumerate(action_names):
        sns.heatmap(q_table[:, idx].reshape(map_size), annot=True, fmt=".2f", ax=axs[idx], cmap='viridis')
        axs[idx].set_title(f'Action: {action}')
    plt.show()



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

    win_count = {i: [] for i in range(num_agents)}
    agent_reward = {i: [] for i in range(num_agents)}
    rewards_per_episode = []
    convergence_time = {i: 0 for i in range(num_agents)}
    max_q_changes = {i: [] for i in range(num_agents)}
    agent_positions = {i: [] for i in range(num_agents)}
    for episode in tqdm(range(episodes)):
        state = env.reset()
        state = state[0]
        occupied_positions = state
        agent_episode_reward = {i: 0 for i in range(num_agents)}
        max_q_change = [0 for i in range(num_agents)]
        all_terminated = False
        reward_until_terminated = 0
        goal_reached = False
        agent_episode_positions = {i: [] for i in range(num_agents)}
        
        for agent, pos in enumerate(occupied_positions):
            agent_episode_positions[agent].append(pos)
        while not all_terminated and not goal_reached:
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
            terminated_agents = info['terminated']
            all_terminated = all(terminated_agents)
            reward_until_terminated += sum(rewards)
            occupied_positions = next_state
            for agent, pos in enumerate(occupied_positions):
                agent_episode_positions[agent].append(pos)
            
            for agent, r in enumerate(rewards):
                if info['goal_reached'][agent]:
                    win_count[agent].append(1)
                    goal_reached = True

            for agent, q in enumerate(q_sa):
                agent_episode_reward[agent] += rewards[agent]
                if block:
                    if agent in info.get("blocking", {}):
                        rewards[agent] += blocking_penalty
                prev_state = q[state[agent], actions[agent]]
                q[state[agent], actions[agent]] = q[state[agent], actions[agent]] + alpha * (rewards[agent] + gamma * np.max(q_sa[agent][next_state[agent]]) - q[state[agent], actions[agent]])
                max_q_change[agent] = max(max_q_change[agent], abs(prev_state - q[state[agent], actions[agent]]))
                
            state = next_state

            for agent in range(num_agents):
               if max_q_change[agent] < 1e-3 and convergence_time[agent] == 0:
                   convergence_time[agent] = episode

        for agent in range(num_agents):
            agent_reward[agent].append(agent_episode_reward[agent])
            q_change = max_q_change[agent]
            max_q_changes[agent].append(q_change)
            agent_positions[agent].append(agent_episode_positions[agent])

        rewards_per_episode.append(reward_until_terminated)
        

    optimal_policy = []
    for agent in range(num_agents):
        optimal_policy.append(np.argmax(q_sa[agent], axis=1))
    return optimal_policy, win_count, agent_reward, convergence_time, max_q_changes, rewards_per_episode, agent_positions
     

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


    optimal_policy, win_count, agent_reward, convergence_time, max_q_changes, rewards_per_episode, agent_positions = agent_training(env, q_sa, gamma, alpha, epsilon, episodes, num_agents, block, blocking_penalty)

    plot_win_count(win_count, num_agents)
    reward_per_agent(agent_reward, num_agents)
    plot_convergence_time(convergence_time, max_q_changes, num_agents)
    plot_reward_trends(rewards_per_episode, num_agents)
    num_rows = env.unwrapped.nrow
    num_cols = env.unwrapped.ncol
    hole_positions = []
    goal_position = None
    for i in range(num_rows):
        for j in range(num_cols):
            if env.unwrapped.desc[i, j] == b'H':
                hole_positions.append(i * num_rows + j)
            elif env.unwrapped.desc[i, j] == b'G':
                goal_position = i * num_rows + j

    plot_trajectories(agent_positions, num_rows, num_cols, hole_positions, goal_position, num_agents, episode=0)
    plot_trajectories(agent_positions, num_rows, num_cols, hole_positions, goal_position, num_agents, episode=499)
    plot_trajectories(agent_positions, num_rows, num_cols, hole_positions, goal_position, num_agents, episode=999)
    

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