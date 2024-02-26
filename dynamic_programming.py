import numpy as np
import matplotlib.pyplot as plt

# python dynamic_programming.py -s -d "SFFHFFF, FHFFFFH, FFFHFFF, HFFFHFG"
# python dynamic_programming.py -s -m "8x8"
# python dynamic_programming.py -s -m "4x4"

# No possible path to the goal
# python dynamic_programming.py -s -d "SFFHFHFF, FHHFFFFH, FFFHFFHF, HFFFHHFG, FFFHFFFF, FHHFHFHF, FFFHFHFF, FFFHFHFF"


def dynamic_programming(env, gamma=0.90, threshold=0.0001):
    """
    Dynamic programming algorithm to solve the Frozen Lake environment

    Args:
    env: OpenAI Gym environment
    gamma: Discount factor
    threshold: Convergence threshold
    
    Returns:
    policy: The optimal policy as an array of integers
    """
    # Initial values
    v_pi = np.zeros(env.observation_space.n)
    # Initil policy
    policy = np.zeros(env.observation_space.n, dtype=int)

    deltas = []  # To track delta per iteration
    value_sums = []  # To track sum of values per iteration

    converged  = False
    step = 0
    while not converged:
        delta = 0
        for state in range(env.observation_space.n):
            current_v = v_pi[state]
            q_pi = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
                    q_pi[action] += probability * reward + probability * gamma * v_pi[next_state]
            q_star = max(q_pi)
            delta = max(delta, abs(current_v - q_star))
            v_pi[state] = q_star

        deltas.append(delta)
        value_sums.append(np.sum(v_pi))

        if delta < threshold:
            converged = True

        step += 1
    print("Converged in {} steps".format(step))

    for state in range(env.observation_space.n):
        q_pi = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for probability, next_state, reward, _ in env.unwrapped.P[state][action]:
                q_pi[action] += probability * reward + probability * gamma * v_pi[next_state]
        policy[state] = np.argmax(q_pi)




     # Graphs
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(deltas, label='Delta per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.title('Convergence of Value Function')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(value_sums, label='Sum of Values')
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Values')
    plt.title('Sum of Values Over Iterations')
    plt.legend()

    plt.tight_layout()
    # Save the graphs
    plt.savefig('dynamic_programming.png')

    return policy

    
