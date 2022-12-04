
"""
Written by Lakshay Sethi

Sharing the code for academic and viewing purposes only!
Prior permission is needed before re-using any part of this code.
Please contact: sethilakshay13@gmail.com for more information.

"""
################################################################################
# Code By Lakshay Sethi                                                        #
# Prior permission is needod before re-using any part of this code             #
################################################################################

import argparse
import numpy as np
from environment import MountainCar, GridWorld


#####################################################################
#Function to collect all Input from the Command Line
#####################################################################
def parse_args() -> tuple:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the probabilirt epsilon for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate/trust parameter
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)

    args = parser.parse_args()
    return args.env, args.mode, args.weight_out, args.returns_out, args.episodes, args.max_iterations, args.epsilon, args.gamma, args.learning_rate


#####################################################################
#Main Function
#####################################################################
if __name__ == "__main__":

    env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()

    #Depending on the user input, instantiating env with an object of MountCar or Gridworld
    if env_type == "mc":
        env = MountainCar(mode)
    elif env_type == "gw":
        env = GridWorld(mode)
    else: raise Exception(f"Invalid environment type {env_type}")

    #Initializing Weights and Rewards Matrices
    weights = np.zeros((env.action_space, 1 + env.state_space), dtype=np.float64)
    rewards = np.zeros((episodes, 1), dtype=np.float64)

    for episode in range(episodes):

        #Getting the current state by calling env.reset()
        currState = env.reset()
        
        #Adding the Bias term
        currState = np.insert(currState, 0, 1)
        episodeReward = 0

        for iteration in range(max_iterations):
            
            #Computing the Quality array
            currQual = np.matmul(weights, currState)

            # Select an action based on the state via the epsilon-greedy strategy
            randNum = np.random.rand()
            if randNum > epsilon:
                actionNum = np.argmax(currQual)
            else:
                actionNum = np.random.randint(env.action_space)

            #Taking a step with the given action to get newState, newReward and doneFlag
            newState, newReward, doneFlag = env.step(actionNum)

            #Adding the Bias term
            newState = np.insert(newState, 0, 1)

            #Computing new Quality after obtaining the newState
            newQual = np.max(np.matmul(weights, newState))

            #Updating the weight
            weights[actionNum] = weights[actionNum] - lr*(currQual[actionNum] - (newReward + gamma*newQual))*currState

            currState = newState
            episodeReward += newReward

            #Breaking out if done
            if doneFlag == True:
                break

        #Update the total rewards for each episode     
        rewards[episode] = episodeReward

    # Save the output weights and rawrds
    np.savetxt(weight_out, weights, fmt="%.18e", delimiter=" ")
    np.savetxt(returns_out, rewards, fmt="%.18e", delimiter=" ")
