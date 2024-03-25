import numpy as np
import matplotlib.pyplot as plt

# python dynamic_programming.py -s -d "SFFHFFF, FHFFFFH, FFFHFFF, HFFFHFG"
# python dynamic_programming.py -s -m "8x8"
# python dynamic_programming.py -s -m "4x4"

# No possible path to the goal
# python dynamic_programming.py -s -d "SFFHFHFF, FHHFFFFH, FFFHFFHF, HFFFHHFG, FFFHFFFF, FHHFHFHF, FFFHFHFF, FFFHFHFF"

def value_iter(env, gamma=0.90, threshold=0.000001, file_name=None):
    # Initial values
    v_pi = np.zeros(env.observation_space.n)
    # Initil policy
    policy = np.zeros(env.observation_space.n, dtype=int)

    deltas = []  
    value_sums = []  

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
    plt.savefig(f'{file_name}.png')
    plt.close()

    return policy, deltas, value_sums, step

def policy_eval(env, experiments):
    """
    Evaluate all the experiments policies based on number of steps to reach the goal, break ties with the sum of values, break ties with convergence steps

    Args:
    env: Frozen Lake environment
    experiments: List of experiments to evaluate

    Returns:
    best_policy: The best policy
    """
    best_policy = None
    best_steps = float('inf')
    best_sum = float('inf')
    best_convergence = float('inf')
    for policy, deltas, value_sums, convergence_steps, gamma, threshold in experiments:
        steps = 0
        for _ in range(100):
            state = env.reset()
            state = state[0]
            terminated = False
            while not terminated:
                state, reward, terminated, truncated, info = env.step(policy[state])
                steps += 1
        if steps < best_steps:
            best_policy = policy
            best_steps = steps
            best_sum = np.sum(value_sums)
            best_convergence = convergence_steps
        elif steps == best_steps:
            sum_values = np.sum(value_sums)
            if sum_values < best_sum:
                best_policy = policy
                best_steps = steps
                best_sum = sum_values
                best_convergence = convergence_steps
            elif sum_values == best_sum:
                if convergence_steps < best_convergence:
                    best_policy = policy
                    best_steps = steps
                    best_sum = sum_values
                    best_convergence = convergence_steps
    return best_policy

def plot_experiment(experiments, file_name='experiment'):
    """
    Plot the experiments results
    """

    plt.figure(figsize=(14, 6))

    for policy, deltas, value_sums, convergence_steps, gamma, threshold in experiments:
        plt.plot(deltas, label=f'Gamma: {gamma}, Threshold: {threshold}')
    plt.xlabel('Iteration')
    plt.ylabel('Delta')
    plt.title('Convergence of Value Function')
    plt.legend()

    plt.tight_layout()
    # Save the graphs
    plt.savefig(f'{file_name}_deltas.png')
    plt.close()

    plt.figure(figsize=(14, 6))

    for policy, deltas, value_sums, convergence_steps, gamma, threshold in experiments:
        plt.plot(value_sums, label=f'Gamma: {gamma}, Threshold: {threshold}')
    plt.xlabel('Iteration')
    plt.ylabel('Sum of Values')
    plt.title('Sum of Values Over Iterations')
    plt.legend()
    
    plt.tight_layout()
    # Save the graphs
    plt.savefig(f'{file_name}_values.png')
    plt.close()
    

def dynamic_programming(env, gamma=0.90, threshold=0.000001, experiment=False):
    """
    Dynamic programming algorithm to solve the Frozen Lake environment

    Args:
    env: OpenAI Gym environment
    gamma: Discount factor
    threshold: Convergence threshold
    
    Returns:
    policy: The optimal policy as an array of integers
    """
    if experiment:
        gammas = [0.7, 0.9, 0.95, 0.99]
        thresholds = [0.00001, 0.000001, 0.0000001]
    else:
        gammas = [gamma]
        thresholds = [threshold]
    experiments = []
    for gamma in gammas:
        for threshold in thresholds:
            file_name = f"dynam_exp/gamma_{gamma}_threshold_{threshold}"
            policy, deltas, value_sums, convergence_steps = value_iter(env, gamma=gamma, threshold=threshold, file_name=file_name)
            experiments.append((policy, deltas, value_sums, convergence_steps, gamma, threshold))

    plot_experiment(experiments, file_name='dynam_exp/experiment')

    best_policy = policy_eval(env, experiments)
    return best_policy

    
