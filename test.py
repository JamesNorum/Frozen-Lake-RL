import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="MultiAgentFrozenLake",
    entry_point='multi_agent_frozen_lake:MultiAgentFrozenLakeEnv',
)

# Ensure your custom environment file is in the Python path or working directory
env = gym.make("MultiAgentFrozenLake", render_mode="human", is_slippery=False, map_name="4x4", agent_positions=[(0,0), (0,1), (0,2)])

import numpy as np

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

# Close the environment when done
env.close()
