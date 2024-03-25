import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

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

def plot_win_count(win_count, num_agents, file_path):
    """
    Plot a histogram of the win count, agents on the x-axis and win count on the y-axis

    Args:
    win_count: The win count
    num_agents: The number of agents

    Returns:
    None
    """
    for agent, wins in win_count.items():
        plt.bar(agent, len(wins), label='Agent ' + str(agent + 1))
    
    plt.xlabel('Agent')
    plt.ylabel('Count')
    plt.title('Win Count')
    plt.legend()
    # Save the plot
    plt.savefig(f'{file_path}/win_count_{num_agents}.png')
    plt.close()
    
def reward_per_agent(reward, num_agents, num_episodes, file_path):
    """
    Plot the average reward per agent

    Args:
    reward: The reward for all agents

    Returns:
    None
    """
    average_reward = {i: [] for i in range(num_agents)}
    skip = num_episodes // 100
    for agent, rewards in reward.items():
        average_reward[agent] = [sum(rewards[i:i+skip])/skip for i in range(0, len(rewards), skip)]

    time = [i for i in range(0, len(reward[0]), skip)]
        
    for agent, rewards in average_reward.items():
        plt.plot(time, rewards, label='Agent ' + str(agent + 1))
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Agent Reward vs Episodes')
    plt.legend()
    # Save the plot
    plt.savefig(f'{file_path}/average_reward_per_agent_{num_agents}.png')
    plt.close()

def plot_convergence_time(q_changes, num_agents, num_episodes, file_path):
    """
    Plot the convergence time

    Args:
    convergence_time: The convergence time

    Returns:
    None
    """
    average_q_changes = {i: [] for i in range(num_agents)}
    # average the q-value changes for each agent over every 100 episodes
    skip = num_episodes // 100
    for agent, q_changes in q_changes.items():
        average_q_changes[agent] = [sum(q_changes[i:i+skip])/skip for i in range(0, len(q_changes), skip)]

    # plot the average q-value changes for each agent over the enire episode count
    # time is the episodes skipped by 100
    time = [i for i in range(0, len(q_changes), skip)]
    for agent, q_changes in average_q_changes.items():
        plt.plot(time, q_changes, label='Agent ' + str(agent + 1))
        
    plt.xlabel('Episodes')
    plt.ylabel('Q-Value Change')
    plt.title('Q-Value Change vs Episodes')
    
    plt.legend(loc='upper right')
    # Save the plot
    plt.savefig(f'{file_path}/q_changes_{num_agents}.png')
    plt.close()

def plot_reward_trends(rewards, num_agents, num_episodes, file_path):
    """
    Plot the reward trend

    Args:
    rewards: The rewards
    num_agents: The number of agents

    Returns:
    None
    """
    skip = num_episodes // 100
    average_reward = [sum(rewards[i:i+skip])/skip for i in range(0, len(rewards), skip)]
    time = [i for i in range(0, len(rewards), skip)]

    plt.figure(figsize=(10, 5))
    plt.plot(time, average_reward, label='Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Trend Over Time')
    plt.legend()
    plt.savefig(f'{file_path}/reward_trend_{num_agents}.png')
    plt.close()



def position_to_coordinates(position, ncols):
    """
    Convert a position in the grid to row and column coordinates.

    Args:
    position: The position in the grid.
    ncols: The number of columns in the grid.

    Returns:
    row: The row coordinate.
    col: The column coordinate.
    """
    return divmod(position, ncols)

def plot_trajectories(agent_positions, nrows, ncols, holes, goal, num_agents, episode, file_path):
    """
    Plot the trajectories of agents, including holes and the goal, with numbered grid squares and padding around the grid.

    Args:
    agent_positions: Dictionary with keys as agent IDs and values as lists of lists of positions for each episode.
    nrows: The number of rows in the grid.
    ncols: The number of columns in the grid.
    holes: A list of positions representing holes in the grid.
    goal: The position of the goal in the grid.
    episode: Optional integer specifying which episode to plot. If None, plots all episodes.

    Returns:
    None
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
    ax.scatter(goal_col, goal_row, color='green', marker='*', s=200, label='Goal')

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
        plt.savefig(f'{file_path}/trajectories_episode_{episode}_{num_agents}.png')
    else:
        plt.savefig(f'{file_path}/trajectories_all_episodes.png')
    plt.close()


def plot_quadrant_heatmap(q_table, nrows, ncols, agent_num, num_agents, file_path, cmap='viridis'):
    """
    Plot a heatmap of the Q-values for each quadrant of the grid.s

    Args:
    q_table: The Q-table.
    nrows: The number of rows in the grid.
    ncols: The number of columns in the grid.
    agent_num: The agent number.
    num_agents: The number of agents.
    file_path: The file path to save the plot.
    cmap: The colormap to use for the heatmap.

    Returns:
    None
    """
    norm = Normalize(vmin=np.min(q_table), vmax=np.max(q_table))
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(figsize=(ncols, nrows))
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()

    for state in range(len(q_table)):
        q_values = q_table[state]
        row, col = divmod(state, ncols)

        
        text_positions = {
            0: (col + 0.2, row + 0.5),  # Left
            1: (col + 0.5, row + 0.8),  # Down
            2: (col + 0.8, row + 0.5),  # Right
            3: (col + 0.5, row + 0.2),  # Up
        }

        
        colors = [mapper.to_rgba(q_value) for q_value in q_values]

        
        quadrants = [
            [(col, row), (col + 0.5, row + 0.5), (col, row + 1)], # Left
            [(col, row + 1), (col + 0.5, row + 0.5), (col + 1, row + 1)], # Down
            [(col + 1, row), (col + 0.5, row + 0.5), (col + 1, row + 1)], # Right
            [(col, row), (col + 0.5, row + 0.5), (col + 1, row)], # Up
        ]

        for idx, vertices in enumerate(quadrants):
            poly = Polygon(vertices, color=colors[idx], ec='k')
            ax.add_patch(poly)
            tx, ty = text_positions[idx]
            ax.text(tx, ty, f'{q_values[idx]:.2f}', ha='center', va='center', color='white', fontsize=9)

    
    ax.set_xticks(np.arange(ncols + 1))
    ax.set_yticks(np.arange(nrows + 1))
    ax.grid(which='major', color='k', linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    plt.title(f'Agent {agent_num + 1} Quadrant Heatmap')
    plt.savefig(f'{file_path}/quadrant_heatmap_agent_{num_agents}_{agent_num}.png')
    plt.close()

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
            agent_reward[agent].append(agent_episode_reward[agent])
            q_change = max_q_change[agent]
            max_q_changes[agent].append(q_change)
            agent_positions[agent].append(agent_episode_positions[agent])

        rewards_per_episode.append(reward_until_terminated)
        

    optimal_policy = []
    for agent in range(num_agents):
        optimal_policy.append(np.argmax(q_sa[agent], axis=1))
    return optimal_policy, win_count, agent_reward, max_q_changes, rewards_per_episode, agent_positions, q_sa
     

def multi_agent_q_learning(agents, episodes, gamma, alpha, epsilon, render_mode, slippery, map_name, desc, block=False, blocking_penalty=-0.1, goal_reward=1, hole_reward=0, step_reward=0):

    env = gym.make("MultiAgentFrozenLake", is_slippery=slippery, map_name=map_name, agent_positions=agents, desc=desc, goal_reward=goal_reward, hole_reward=hole_reward, step_reward=step_reward)

    num_agents = len(agents)

    if not os.path.exists(f'multi_figures/{agents}_{num_agents}'):
        os.makedirs(f'multi_figures/{agents}_{num_agents}')

    file_path = f'multi_figures/{agents}_{num_agents}'

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


    optimal_policy, win_count, agent_reward, max_q_changes, rewards_per_episode, agent_positions, q_table = agent_training(env, q_sa, gamma, alpha, epsilon, episodes, num_agents, block, blocking_penalty)

    plot_win_count(win_count, num_agents, file_path)
    reward_per_agent(agent_reward, num_agents, episodes, file_path)
    plot_convergence_time(max_q_changes, num_agents, episodes, file_path)
    plot_reward_trends(rewards_per_episode, num_agents, episodes, file_path)
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

    # list of episodes to plot, the first, middle and last episode
    plot_episodes = [0, episodes//2, episodes-1]
    for episode in plot_episodes:
        plot_trajectories(agent_positions, num_rows, num_cols, hole_positions, goal_position, num_agents, episode, file_path)


    for agent in range(num_agents):
        plot_quadrant_heatmap(q_table[agent], num_rows, num_cols, agent, num_agents, file_path)
    

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