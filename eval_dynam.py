"""
This code was adapted from:
https://aleksandarhaber.com/iterative-policy-evaluation-algorithm-in-python-reinforcement-learning-tutorial/

The only changes made were to adapt it to my env, add graphs, and return the policy.
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaldyn(env):
    env.reset()

    # observation space - states 
    env.observation_space
    
    # actions: left -0, down - 1, right - 2, up- 3
    env.action_space
    
    
    #transition probabilities
    #p(s'|s,a) probability of going to state s' 
    #          starting from the state s and by applying the action a
    
    # env.P[state][action]
    env.unwrapped.P[0][1] #state 0, action 1
    # output is a list having the following entries
    # (transition probability, next state, reward, Is terminal state?)

    # select the discount factor
    discountFactor=0.9
    # initialize the value function vector
    valueFunctionVector=np.zeros(env.observation_space.n)
    # maximum number of iterations
    maxNumberOfIterations=1000
    # convergence tolerance delta
    convergenceTolerance=10**(-6)
    
    # convergence list 
    convergenceTrack=[]

    policy = np.zeros(env.observation_space.n, dtype=int)
    deltas = []  # To track delta per iteration
    value_sums = []  # To track sum of values per iteration
    step = 0
    
    for iterations in range(maxNumberOfIterations):
        convergenceTrack.append(np.linalg.norm(valueFunctionVector,2))
        valueFunctionVectorNextIteration=np.zeros(env.observation_space.n)
        delta = 0
        for state in env.unwrapped.P:
            outerSum=0
            current_v = valueFunctionVector[state]
            q_pi = np.zeros(env.action_space.n)
            for action in env.unwrapped.P[state]:
                innerSum=0
                for probability, nextState, reward, isTerminalState in env.unwrapped.P[state][action]:
                    #print(probability, nextState, reward, isTerminalState)
                    q_pi[action] = innerSum + probability*(reward+discountFactor*valueFunctionVector[nextState])
                    innerSum=innerSum+ probability*(reward+discountFactor*valueFunctionVector[nextState])

                outerSum=outerSum+0.25*innerSum
            q_star = max(q_pi)
            delta = max(delta, abs(current_v - q_star))
            valueFunctionVectorNextIteration[state]=outerSum
        if(np.max(np.abs(valueFunctionVectorNextIteration-valueFunctionVector))<convergenceTolerance):
            valueFunctionVector=valueFunctionVectorNextIteration
            print(f'Converged in {step} steps')
            break
        deltas.append(delta)
        value_sums.append(np.sum(valueFunctionVectorNextIteration))
        valueFunctionVector=valueFunctionVectorNextIteration   
        step += 1  

    for state in range(env.observation_space.n):
        q_pi = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for probability, next_state, reward, _ in env.unwrapped.P[state][action]:
                q_pi[action] += probability * reward + probability * discountFactor * valueFunctionVector[next_state]
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
    plt.savefig('eval_dynamic_programming.png')

    return policy
