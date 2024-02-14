import gymnasium as gym 

# Constants for the FrozenLake environment
IS_SLIPPERY = False
MAP_NAME = '4v4'
DESC = ["SFFF"]
RENDER_MODE = 'rgb_array'

# Create the FrozenLake environment
env = gym.make('FrozenLake-v1', is_slippery=IS_SLIPPERY, map_name=MAP_NAME, desc=DESC, render_mode=RENDER_MODE)


