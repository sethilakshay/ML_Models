
"""
Written by Lakshay Sethi

Sharing the code for academic and viewing purposes only!
Prior permission is needed before re-using any part of this code.
Please contact: sethilakshay13@gmail.com for more information.

"""

import sys
import numpy as np

def majority_vote_classifier(infile, outfile):
    #Reading in the data
    data_train = np.genfromtxt(infile, delimiter="\t", dtype=None, encoding=None)
    data_test  = np.genfromtxt(outfile, delimiter="\t", dtype=None, encoding=None)

    #Slicing the data removing header columns
    data_train = data_train[1:]
    data_test = data_test[1:]

    #Using the Majority vote Classifier method (Prediction value is between 1 and 0)
    def predict (data, col_num):
        cnt_0 = 0
        cnt_1 = 1

        for i in range(len(data)):
            if (data[i, col_num] == 0):
                cnt_0 += 1
            else:
                cnt_1 += 1
                
        if (cnt_1 > cnt_0):
            return 1
        elif (cnt_0 > cnt_1):
            return 0
        else:
            predict(data, col_num-1)

    #Using the last column to run the majority classifier
    prediction = predict(data_train, -1)

    #Calculating the training error rate
    def error_cal (data, prediction):
        res = 0
        for i in range(len(data)):
            if(int(data[i,-1]) != prediction):
                res += 1
        return (res/len(data))

    error_train = error_cal (data_train, prediction)
    error_test = error_cal (data_test, prediction)

    #Creating Arrays to export the final labels and metrics as txt files
    train_labels = np.full((len(data_train), 1), prediction)
    test_labels = np.full((len(data_test), 1), prediction)
    metrics = np.array([["error(train):", "%f"%round(error_train,6)], ["error(test):", "%f"%round(error_test,6)]], dtype=object)

    np.savetxt('train.txt', train_labels, fmt = "%s")
    np.savetxt('test.txt', test_labels, fmt = "%s")
    np.savetxt('metrics.txt', metrics, fmt = "%s")

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    print("The input file is: %s" % (infile))
    print("The output file is: %s" % (outfile))

    #Calling the Majority Vote Classifier function
    majority_vote_classifier(infile, outfile)
