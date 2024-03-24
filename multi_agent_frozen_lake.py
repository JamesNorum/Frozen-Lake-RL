from contextlib import closing
from io import StringIO
from os import path
from typing import List, Optional
import pygame.font

import numpy as np

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import seeding


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


# DFS to check that it's a valid path.
def is_valid(board: List[List[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(
    size: int = 8, p: float = 0.8, seed: Optional[int] = None
) -> List[str]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: optional seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]


class MultiAgentFrozenLakeEnv(Env):
    """
    Frozen lake involves crossing a frozen lake from start to goal without falling into any holes
    by walking over the frozen lake.
    The player may not always move in the intended direction due to the slippery nature of the frozen lake.

    ## Description
    The game starts with the player at location [0,0] of the frozen lake grid world with the
    goal located at far extent of the world e.g. [3,3] for the 4x4 environment.

    Holes in the ice are distributed in set locations when using a pre-determined map
    or in random locations when a random map is generated.

    The player makes moves until they reach the goal or fall in a hole.

    The lake is slippery (unless disabled) so the player may move perpendicular
    to the intended direction sometimes (see <a href="#is_slippy">`is_slippery`</a>).

    Randomly generated worlds will always have a path to the goal.

    Elf and stool from [https://franuka.itch.io/rpg-snow-tileset](https://franuka.itch.io/rpg-snow-tileset).
    All other assets by Mel Tillery [http://www.cyaneus.com/](http://www.cyaneus.com/).

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up

    ## Observation Space
    The observation is a value representing the player's current position as
    current_row * nrows + current_col (where both the row and col start at 0).

    For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
    The number of possible observations is dependent on the size of the map.

    The observation is returned as an `int()`.

    ## Starting State
    The episode starts with the player in state `[0]` (location [0, 0]).

    ## Rewards

    Reward schedule:
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0

    ## Episode End
    The episode ends if the following happens:

    - Termination:
        1. The player moves into a hole.
        2. The player reaches the goal at `max(nrow) * max(ncol) - 1` (location `[max(nrow)-1, max(ncol)-1]`).

    - Truncation (when using the time_limit wrapper):
        1. The length of the episode is 100 for 4x4 environment, 200 for FrozenLake8x8-v1 environment.

    ## Information

    `step()` and `reset()` return a dict with the following keys:
    - p - transition probability for the state.

    See <a href="#is_slippy">`is_slippery`</a> for transition probability information.


    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    ```

    `desc=None`: Used to specify maps non-preloaded maps.

    Specify a custom map.
    ```
        desc=["SFFF", "FHFH", "FFFH", "HFFG"].
    ```

    A random generated map can be specified by calling the function `generate_random_map`.
    ```
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map

    gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
    ```

    `map_name="4x4"`: ID to use any of the preloaded maps.
    ```
        "4x4":[
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
            ]

        "8x8": [
            "SFFFFFFF",
            "FFFFFFFF",
            "FFFHFFFF",
            "FFFFFHFF",
            "FFFHFFFF",
            "FHHFFFHF",
            "FHFFHFHF",
            "FFFHFFFG",
        ]
    ```

    If `desc=None` then `map_name` will be used. If both `desc` and `map_name` are
    `None` a random 8x8 map with 80% of locations frozen will be generated.

    <a id="is_slippy"></a>`is_slippery=True`: If true the player will move in intended direction with
    probability of 1/3 else will move in either perpendicular direction with
    equal probability of 1/3 in both directions.

    For example, if action is left and is_slippery is True, then:
    - P(move left)=1/3
    - P(move up)=1/3
    - P(move down)=1/3


    ## Version History
    * v1: Bug fixes to rewards
    * v0: Initial version release

    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 3,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
        agent_positions=[(0, 0), (1, 0)], 
        goal_reward=1, 
        hole_reward=0, 
        step_reward=0
    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.agent_positions = agent_positions
        self.reset_positions = agent_positions
        self.lastaction = None
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.is_slippery = is_slippery
        self.terminated_agents = [False for _ in self.agent_positions]
        # Initialize font; you may choose another font and size
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 24)

        self.goal_reward = goal_reward
        self.hole_reward = hole_reward
        self.step_reward = step_reward



        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"GH"
            reward = float(newletter == b"G")
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        
        num_agents = len(self.agent_positions)

        self.observation_space = spaces.Tuple([spaces.Discrete(self.nrow * self.ncol) for _ in range(num_agents)])
        self.action_space = spaces.Tuple([spaces.Discrete(4) for _ in range(num_agents)])
        self.observation_space.n = nS
        self.action_space.n = nA 

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None



    def calculate_new_position(self, current_position, action, is_slippery):
        import random
        if is_slippery:
            # Define perpendicular actions
            if action == LEFT:
                possible_actions = [LEFT, DOWN, UP]
            elif action == DOWN:
                possible_actions = [LEFT, DOWN, RIGHT]
            elif action == RIGHT:
                possible_actions = [DOWN, RIGHT, UP]
            elif action == UP:
                possible_actions = [LEFT, RIGHT, UP]
            
            # Choose an action with slipperiness taken into account
            action = random.choice(possible_actions)
        
        # Original movement calculation
        row, col = current_position
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.nrow - 1)
        elif action == RIGHT:
            col = min(col + 1, self.ncol - 1)
        elif action == UP:
            row = max(row - 1, 0)
        
        return (row, col)


    def step(self, actions):
        assert len(actions) == len(self.agent_positions), "Each agent needs an action."

        self.lastaction = actions  # For rendering

        rewards = [0 for _ in self.agent_positions]  # Initialize rewards for each agent
        terminated = [False for _ in self.agent_positions]  # Initialize done flags for each agent
        info = {}  # Placeholder for additional info
        info["goal_reached"] = {i: None for i in range(len(self.agent_positions))}

        # Calculate new positions without updating yet
        new_positions = []
        # Inside your step function
        for agent_idx, action in enumerate(actions):
            if self.terminated_agents[agent_idx]:
                new_positions.append(self.agent_positions[agent_idx])
                continue
            current_position = self.agent_positions[agent_idx]
            new_position = self.calculate_new_position(current_position, action, self.is_slippery) # Pass is_slippery here
            new_positions.append(new_position)


        # Create a temporary list to track occupied positions for collision resolution
        occupied_positions = set(self.agent_positions)  # Start with current positions to allow moving away
        # update the occupied positions to only include non-terminated agents
        occupied_positions = set([pos for idx, pos in enumerate(self.agent_positions) if not self.terminated_agents[idx]])
        collisions_resolved_positions = []
        for idx, pos in enumerate(new_positions):
            if pos in occupied_positions:
                # Collision detected, revert to original position if new position is already occupied
                collisions_resolved_positions.append(self.agent_positions[idx])
            else:
                # No collision, update occupied positions and move to the new position
                occupied_positions.add(pos)
                collisions_resolved_positions.append(pos)

        # Update agent positions after resolving collisions
        self.agent_positions = collisions_resolved_positions

        # Before updating states and rewards based on new positions
        blocking_info = {}  # Placeholder for blocking info (agent_index: bool)
        for i, pos_i in enumerate(new_positions):
            for j, pos_j in enumerate(new_positions):
                if i != j and pos_i == pos_j:
                    # Example condition for blocking, can be refined
                    blocking_info[i] = True  # Mark agent i as involved in a blocking scenario

        # Pass blocking info to the info dictionary to use it later in training
        info["blocking"] = blocking_info


        # Update states and rewards based on the new positions
        states = []
        for agent_idx, position in enumerate(self.agent_positions):
            row, col = position
            state = self.to_s(row, col)
            states.append(state)

            # Determine the reward and termination
            letter = self.desc[row, col]
            if letter == b'G':
                rewards[agent_idx] = self.goal_reward  # Reward for reaching the goal
                terminated[agent_idx] = True
                info["winning_agent"] = agent_idx
                info["goal_reached"] = {i: True if i == agent_idx else False for i in range(len(self.agent_positions))}
            elif letter == b'H':
                rewards[agent_idx] = self.hole_reward  # Reward for falling into a hole
                terminated[agent_idx] = True
            else:
                rewards[agent_idx] = self.step_reward  # Standard reward for non-terminal state

            # Update terminated agents list
            if terminated[agent_idx]:
                self.terminated_agents[agent_idx] = True

        any_terminated = any(terminated)
        
        # Combine the states of all agents into a single state
        next_state = self.get_combined_state()
        self.s = states  # Assuming self.s is meant to track states of all agents

        total_reward = sum(rewards)

        info["rewards"] = rewards
        info["states"] = states
        info["terminated"] = terminated
        info["occupied_positions"] = occupied_positions

        return next_state, total_reward, any_terminated, False, info


    def get_combined_state(self):
        """Combine agents' positions into a single state representation."""
        combined_state = []
        for row, col in self.agent_positions:
            combined_state.append(row * self.ncol + col)
        return tuple(combined_state)

    def to_s(self, row, col):
        """Converts a (row, col) pair to a single integer representing the state."""
        return row * self.ncol + col


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)  # It's good practice to call super().reset()
        
        # Reset or initialize agent positions
        self.agent_positions = self.reset_positions
        # Set for the number of agents
        self.s = [self.to_s(*pos) for pos in self.agent_positions]
        self.terminated_agents = [False for _ in self.agent_positions]

        # Create initial observations based on agent positions
        initial_obs = [self.to_s(*pos) for pos in self.agent_positions]
        
        return tuple(initial_obs), {"prob": 1}  # Return observations and initial probability info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            return self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)
        
    def _render_gui(self, mode):
        import pygame
        from os import path

        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Frozen Lake")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/hole.png")
            self.hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.cracked_hole_img is None:
            file_name = path.join(path.dirname(__file__), "img/cracked_hole.png")
            self.cracked_hole_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.ice_img is None:
            file_name = path.join(path.dirname(__file__), "img/ice.png")
            self.ice_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(path.dirname(__file__), "img/goal.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.start_img is None:
            file_name = path.join(path.dirname(__file__), "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.elf_images is None:
            elfs = [
                path.join(path.dirname(__file__), "img/elf_left.png"),
                path.join(path.dirname(__file__), "img/elf_down.png"),
                path.join(path.dirname(__file__), "img/elf_right.png"),
                path.join(path.dirname(__file__), "img/elf_up.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in elfs
            ]

        desc = self.desc.tolist()
        assert isinstance(desc, list), f"desc should be a list or an array, got {desc}"
        for y in range(self.nrow):
            for x in range(self.ncol):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = (*pos, *self.cell_size)

                self.window_surface.blit(self.ice_img, pos)
                if desc[y][x] == b"H":
                    self.window_surface.blit(self.hole_img, pos)
                elif desc[y][x] == b"G":
                    self.window_surface.blit(self.goal_img, pos)
                elif desc[y][x] == b"S":
                    self.window_surface.blit(self.start_img, pos)

                pygame.draw.rect(self.window_surface, (180, 200, 230), rect, 1)
                
        # Render each agent
        for agent_idx, (agent_row, agent_col) in enumerate(self.agent_positions):
            pos = (agent_col * self.cell_size[0], agent_row * self.cell_size[1])
            last_action = self.lastaction[agent_idx] if self.lastaction is not None else 1
            # Check if the agent is on a hole and has terminated
            if self.terminated_agents[agent_idx] and self.desc[agent_row][agent_col] == b'H':
                self.window_surface.blit(self.cracked_hole_img, pos)
            # Render the elf on the goal or any other tile
            else:
                self.window_surface.blit(self.elf_images[last_action], pos)

            # Render the agent number. Adjust the position as needed.
            agent_text = self.font.render(str(agent_idx + 1), True, (0, 0, 0))
            text_pos = pos[0] + 5, pos[1] + 5  # Adjust as needed
            self.window_surface.blit(agent_text, text_pos)


        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2))

    @staticmethod
    def _center_small_rect(big_rect, small_dims):
        offset_w = (big_rect[2] - small_dims[0]) / 2
        offset_h = (big_rect[3] - small_dims[1]) / 2
        return (
            big_rect[0] + offset_w,
            big_rect[1] + offset_h,
        )

    def _render_text(self):
        desc = self.desc.tolist()
        outfile = StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.lastaction]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()