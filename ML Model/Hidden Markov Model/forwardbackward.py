
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
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file


#####################################################################
#Function to compute the Log Likelihood using LSE (LogSumExp)
#####################################################################
def calcLogLikelihood(alpha_final: np.array, n: int) -> float:

    #To prevent underflow in Log-Sum exponential technique
    max_alpha_final = np.max(alpha_final)
    exp_alpha_final = np.exp(alpha_final - max_alpha_final)
    
    return (np.log(exp_alpha_final.sum())+max_alpha_final)/n


#####################################################################
#Function to calculate the accurte predictions
#####################################################################
def calcAccuracy(validation_obs: list, predictions: np.array, hiddenState_data: dict) -> int:

    actual_y = np.transpose(np.array(validation_obs))[1]
    actual_y_indices = np.array([hiddenState_data[i] for i in actual_y])

    return np.array(actual_y_indices == predictions).sum()    


#####################################################################
#Function to run the Forward algorithim of HMM
#####################################################################
def forward(validation_obs: list, initalState_log: np.array, transState_log: np.array, emitState_log: np.array, words_to_indices: dict) -> np.array:
    
    T = len(validation_obs)
    log_alpha = np.zeros(shape = (T, len(initalState_log)))
    
    #Base Case
    idx = words_to_indices[validation_obs[0][0]]
    log_alpha[0] = initalState_log + np.transpose(emitState_log)[idx]

    for i in range(1, T):
        idx = words_to_indices[validation_obs[i][0]]

        #To prevent underflow in Log-Sum exponential technique
        max_alpha = np.max(log_alpha[i-1])
        max_trans = np.max(transState_log)

        exp_alpha = np.exp(log_alpha[i-1] - max_alpha)
        exp_trans = np.exp(transState_log - max_trans)

        log_alpha[i] =  np.transpose(emitState_log)[idx] + np.log(np.dot(exp_alpha, exp_trans)) + max_alpha + max_trans
        
    return log_alpha


#####################################################################
#Function to run the Backward algorithim of HMM
#####################################################################
def backward(validation_obs: list, initalState_log: np.array, transState_log: np.array, emitState_log: np.array, words_to_indices: dict) -> np.array:

    T = len(validation_obs)
    log_beta = np.zeros(shape = (T, len(initalState_log)))

    #Base Case
    #No need to inialzie log_beta[T-1] since log_beta is alread 0 initialized 

    for i in range(T-2, -1, -1):
        idx = words_to_indices[validation_obs[i+1][0]]
        currEmit = np.transpose(emitState_log)[idx]
        currEmit_Beta = currEmit + log_beta[i+1]

        #To prevent underflow in Log-Sum exponential technique
        max_currEmit_Beta = np.max(currEmit_Beta)
        max_trans = np.max(transState_log)

        exp_currEmit_Beta = np.exp(currEmit_Beta - max_currEmit_Beta)
        exp_trans = np.exp(transState_log - max_trans)
        
        log_beta[i] = np.log(np.dot(exp_trans, exp_currEmit_Beta)) + max_trans + max_currEmit_Beta

    return log_beta


#####################################################################
#Main Function
#####################################################################
if __name__ == "__main__":

    # Collecting the input datasets from command line
    validation_data, words_to_indices, tags_to_indices, init, emit, trans, outPredicted, outMetrics = get_inputs()

    indices_to_tags = dict()
    for key, value in tags_to_indices.items():
        indices_to_tags[value] = key

    #Initialzing variables
    n = len(validation_data)

    matrix_alpha_All = list()
    matrix_beta_All = list()

    avgLogLikelihood = 0
    correctPredictions_cnt = 0
    totWords = 0

    predicted_vals_All = list()

    #Running the foward algorithim to make predictions
    for i in range(n):
        #Computing the Alpha matrix
        matrix_alpha_All.append(forward(validation_data[i], np.log(init), np.log(trans), np.log(emit), words_to_indices))
        avgLogLikelihood += calcLogLikelihood(matrix_alpha_All[i][len(validation_data[i])-1], n)

        #Computing the Beta matrix
        matrix_beta_All.append(backward(validation_data[i], np.log(init), np.log(trans), np.log(emit), words_to_indices))

        predictions_idx = np.argmax(matrix_alpha_All[i] + matrix_beta_All[i], axis = 1)

        correctPredictions_cnt += calcAccuracy(validation_data[i], predictions_idx, tags_to_indices)
        totWords += len(validation_data[i])

        predictions_vals = [indices_to_tags[idx] for idx in predictions_idx]
        predicted_vals_All.append(predictions_vals)

    #Computing the accuracy of 1st Markov Assumption HMM
    accuracy = correctPredictions_cnt/totWords

    #Outputing Metrics and Predictions
    with open(outMetrics, "w") as f:
        f.write("Average Log-Likelihood:" + " " + str(avgLogLikelihood) + "\n")
        f.write("Accuracy:" + " " + str(accuracy))

    with open(outPredicted, "w") as f:

        for i in range(len(validation_data)):
            for j in range(len(validation_data[i])):
                f.write(validation_data[i][j][0] + "\t" + predicted_vals_All[i][j])
                f.write("\n")
            f.write("\n")