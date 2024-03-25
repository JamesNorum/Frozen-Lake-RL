import gymnasium as gym
import pygame
import sys

# Initialize Pygame for capturing keyboard inputs
pygame.init()

# Set up a dummy display to capture events
screen = pygame.display.set_mode((1, 1))
pygame.display.set_caption("Control Frozen Lake with Pygame")

# Map the arrow keys to actions
key_action_mapping = {
    pygame.K_LEFT: 0,
    pygame.K_DOWN: 1,
    pygame.K_RIGHT: 2,
    pygame.K_UP: 3,
}

def play(env):
    state = env.reset()
    env.render()  # Initial render of the game state
    done = False

    while not done:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in key_action_mapping:
                    # Perform action in the environment based on key press
                    action = key_action_mapping[event.key]
                    state, reward, done, _, _ = env.step(action)
                    env.render()  # Render the updated game state
                    print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")

                # Restart the game if it has ended
                if done:
                    print("Game Over. Press any key to restart or close the window to exit.")
                    restart_wait = True
                    while restart_wait:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                state = env.reset()
                                env.render()
                                done = False
                                restart_wait = False
                            elif event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit()

if __name__ == "__main__":
    environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
    try:
        play(environment)
    finally:
        pygame.quit()
