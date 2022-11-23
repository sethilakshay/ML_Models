
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


#####################################################################
#Function to collect all Input from the Command Line
#####################################################################
def get_inputs():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans


#####################################################################
#Function to obtain Initial Matrix for the HMM Model
#####################################################################
def matrixInitial(train_data: list, hiddenState_data: dict) -> np.array:
    
    #Initalzing the array to 1's to account for cases where a state might be zero
    initialList = np.ones(shape = (len(hiddenState_data)))
    

    for i in range(len(train_data)):
        #Here train_data[i][0][1] will give the hidden state value at the 1st instance
        init_state_val = train_data[i][0][1]
        idx = hiddenState_data[init_state_val]
        initialList[idx] += 1

    #Normalizing Initial arr to get probabilities the states
    initialList = initialList/initialList.sum()

    return initialList


#####################################################################
#Function to obtain Transition Matrix for the HMM Model
#####################################################################
def matrixTransition(train_data: list, hiddenState_data: dict) -> np.array:
    
    #Initalzing the array to 1's to account for cases where a state might be zero
    transitionArr = np.ones(shape = (len(hiddenState_data), len(hiddenState_data)))

    for i in range(len(train_data)):

        for j in range(len(train_data[i])-1):

            x_idx_state = train_data[i][j][1]
            y_idx_state = train_data[i][j+1][1]

            transitionArr[hiddenState_data[x_idx_state]][hiddenState_data[y_idx_state]] += 1 

    #Normalizing transition arr to get probabilities for each row
    transitionArr = transitionArr/transitionArr.sum(axis = 1).reshape(-1,1)
    return transitionArr


#####################################################################
#Function to obtain Emission Matrix for the HMM Model
#####################################################################
def matrixEmission(train_data: list, hiddenState_data: dict, obsState_data: dict) -> np.array:

    #Initalzing the array to 1's to account for cases where a state might be zero
    emissionArr = np.ones(shape = (len(hiddenState_data), len(obsState_data)))

    for i in range(len(train_data)):

        for j in range(len(train_data[i])):

            x_idx_state = train_data[i][j][1]
            y_idx_state = train_data[i][j][0]

            emissionArr[hiddenState_data[x_idx_state]][obsState_data[y_idx_state]] += 1

    #Normalizing transition arr to get probabilities for each row
    emissionArr = emissionArr/emissionArr.sum(axis = 1).reshape(-1,1)
    return emissionArr


#####################################################################
#Main Function
#####################################################################
if __name__ == "__main__":
    
    # Collecting the input datasets from command line
    train_data, words_to_indices, tags_to_indices, outInitial, outEmission, outTransmision = get_inputs()

    # Initializing the initial, emission, and transition matrices
    init = matrixInitial(train_data, tags_to_indices)
    trans = matrixTransition(train_data, tags_to_indices)
    emit = matrixEmission(train_data, tags_to_indices, words_to_indices)

    # Outputing Initial, Transition and Emission matrices
    np.savetxt(outInitial, init, fmt="%s")
    np.savetxt(outTransmision, trans, fmt="%s")
    np.savetxt(outEmission, emit, fmt="%s")