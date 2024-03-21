import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="MultiAgentFrozenLake",
    entry_point='multi_agent_frozen_lake:MultiAgentFrozenLakeEnv',
)

def multi_agent_q_learning(agents, episodes, gamma, alpha, epsilon, render_mode, slippery, map_name, desc):
    # Ensure your custom environment file is in the Python path or working directory
    env = gym.make("MultiAgentFrozenLake", is_slippery=slippery, map_name=map_name, agent_positions=agents, desc=desc)

    # Initialize the environment
    env.reset()

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

        
    # Number of episodes to run
    win = False
    env = gym.make("MultiAgentFrozenLake", render_mode=render_mode, is_slippery=slippery, map_name=map_name, agent_positions=agents, desc=desc)
    env.reset()
    while not win:
        # Choose a random action
        actions = env.action_space.sample()
        #actions = (0, 2, 1)
        
        print_action(actions)
        print("")
        
        # Execute the action
        next_state, reward, terminated, truncated, info  = env.step(actions)

        print(reward, info)

        # Render the environment
        env.render()
        
        print(terminated)

        if terminated:
            win = True
            print("Game Over")
            break

    return None