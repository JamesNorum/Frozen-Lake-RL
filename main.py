import argparse
import gymnasium as gym
# Import the algorithms
from dynamic_programming import dynamic_programming
from q_learning import q_learning
from monte_carlo import every_visit_monte_carlo, first_visit_monte_carlo
from eligibility_traces import sarsa_lambda_et

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
    for i in range(0, len(optimal_policy), 4):
        # print arrows for the optimal policy
        # print the H for the hole and G for the goal in place of the arrow
        for j in range(i, i+4):
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

def main():
    """
    Main function to run the Frozen Lake environment with different algorithms

    Command Line Arguments:
    -al, --algorithm: The algorithm to use for solving the Frozen Lake environment. Default is dynamic. The options are d for Dynamic, q for Q-Learning, m for Monte Carlo, and e for Eligibility Traces.
    -g, --gamma: Discount factor gamma.
    -t, --threshold: Convergence threshold.
    -a, --alpha: Alpha Value.
    -e, --epsilon: Epsilon Value.
    -ep, --episodes: Number of episodes.
    -d, --desc: Description for the Frozen Lake environment.
    -m, --map: Map size for the Frozen Lake environment.
    -r, --render: Render mode for the Frozen Lake environment.
    -s, --slippery: Use slippery mode, Default is false.

    Output:
    The optimal policy for the Frozen Lake environment

    Display:
    The Frozen Lake environment with the optimal policy
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser(description='Dynamic programming for Frozen Lake environment.')
    parser.add_argument('-al', '--algorithm', type=str, default="d", choices={"d", "q", "mcfv", "mcev", "et"}, help='The algorithm to use for solving the Frozen Lake environment. Default is dynamic. The options are d for Dynamic, q for Q-Learning, mcfv for Monte Carlo: First Visit, mcev for Monte Carlo: Every Visit, and e for Eligibility Traces.')
    parser.add_argument('-g', '--gamma', type=float, default=0.90, help='Discount factor gamma.')    
    parser.add_argument('-t', '--threshold', type=float, default=0.0001, help='Convergence threshold.')
    parser.add_argument('-a', '--alpha', type=float, default=0.08, help='Alpha Value.')
    parser.add_argument('-e', '--epsilon', type=float, default=0.1, help='Epsilon Value.')
    parser.add_argument('-ep', '--episodes', type=int, default=5000, help='Number of episodes.')
    parser.add_argument('-l', '--lambda_', type=float, default=0.9, help='Lambda Value. Used for eligibility trace decay')
    parser.add_argument('-d', '--desc', type=str, default=None, help='Description for the Frozen Lake environment.')
    parser.add_argument('-m', '--map', type=str, default="4x4", help='Map size for the Frozen Lake environment.')
    parser.add_argument('-r', '--render', type=str, default="human", help='Render mode for the Frozen Lake environment.')
    parser.add_argument('-s', '--slippery', action='store_true', help='Use slippery mode, Default is false.')
    args = parser.parse_args()

    # Store the arguments
    algorithm = args.algorithm
    gamma = args.gamma
    epsilon = args.epsilon
    alpha = args.alpha
    threshold = args.threshold
    episodes = args.episodes
    lambda_ = args.lambda_
    desc = args.desc
    if desc is not None:
        desc = desc.split(",")
        desc = list(map(str, desc))
        desc = [x.strip() for x in desc]
    if desc is None:
        map_name = args.map
    else:
        map_name = "custom"
    # Check if the map is 4x4 or 8x8
    if map_name not in ["4x4", "8x8", "custom"]:
        raise ValueError("Map Argument -m : Map size should be 4x4, 8x8, or custom.")
    render_mode = args.render
    slippery = args.slippery

    print(f"Running Frozen Lake with parameters: Algorithm: {algorithm}, Gamma: {gamma}, Epsilon: {epsilon}, Alpha: {alpha}, Threshold: {threshold}, Episodes: {episodes}, Description: {desc}, Map: {map_name}, Render Mode: {render_mode}, Slippery: {slippery}")    

    # Make environment according to arguments
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=slippery)
    if algorithm == "d":
        print("Dynamic Programming")
        optimal_policy = dynamic_programming(env=env, gamma=gamma, threshold=threshold)
    elif algorithm == "q":
        print("Q-Learning")
        optimal_policy = q_learning(env=env, episodes=episodes, gamma=gamma, alpha=alpha, epsilon=epsilon)
    elif algorithm == "mcfv":
        print("Monte Carlo: First Visit")
        optimal_policy = first_visit_monte_carlo(env=env, episodes=episodes, gamma=gamma, epsilon=epsilon)
    elif algorithm == "mcev":
        print("Monte Carlo: Every Visit")
        optimal_policy = every_visit_monte_carlo(env=env, episodes=episodes, gamma=gamma, epsilon=epsilon)
    elif algorithm == "et":
        print("Eligibility Traces")
        optimal_policy = sarsa_lambda_et(env=env, episodes=episodes, gamma=gamma, alpha=alpha, epsilon=epsilon, lambda_=lambda_)

    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    TODO
    JAMES:
        Add other algorithms here once implemented
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """ 


    print("Optimal policy:\n", optimal_policy)
    
    print_optimal_policy(optimal_policy, env)

    # Change render mode to human
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=slippery, render_mode=render_mode)
    observation, info = env.reset()
    win = False
    timestep = 0
    while not win:
        env.render()
        action = optimal_policy[observation]
        observation, reward, terminated, truncated, info = env.step(action)
        # print(observation, reward, terminated, truncated, info, action)
        if reward:
            print("Reward: ", reward)
            win = True
        # Check if in a hole
        if env.unwrapped.desc.flat[observation] == b'H':
            print("Fell in a hole")
        if terminated:
            print("Episode finished after {} timesteps".format(timestep+1))
            observation, info = env.reset()

        timestep += 1

    env.close()

if __name__ == "__main__":
    main()