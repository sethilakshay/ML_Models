
"""
Written by Lakshay Sethi

Sharing the code for academic and viewing purposes only!
Prior permission is needed before re-using any part of this code.
Please contact: sethilakshay13@gmail.com for more information.

"""

import sys
import numpy as np

#This function will calculate the Entropy and Majority value of the label
def predict_calc (data, col_num):
    n = len(data)
    cnt_0 = 0
    cnt_1 = 0
    #Majority Vote Classifier
    for i in range(n):
        if (int(data[i, col_num]) == 0):
            cnt_0 += 1
        else:
            cnt_1 += 1
    if (cnt_1 > cnt_0):
        predict_val = 1
    elif (cnt_0 > cnt_1):
        predict_val = 0
    else:
        predict_val = 1

    #Calculating Entropy
    if (cnt_0 == 0):
        entrpy = -((cnt_1/n)*np.log2(cnt_1/n))
    elif (cnt_1 == 0):
        entrpy = -((cnt_0/n)*np.log2(cnt_0/n))
    else:
        entrpy = -((cnt_0/n)*np.log2(cnt_0/n)) -((cnt_1/n)*np.log2(cnt_1/n)) 

    return predict_val, entrpy

def error_cal (data, prediction):
    res = 0
    for i in range(len(data)):
        if(int(data[i,-1]) != prediction):
            res += 1
    return (res/len(data))

def func(infile, outfile):
    #Reading and Slicing the data
    data_train = np.genfromtxt(infile, delimiter="\t", dtype=None, encoding=None)
    data_train = data_train[1:]

    #Calculting the entropy and most common value of the label
    prediction, entropy = predict_calc(data_train, -1)

    #Calculating the training error rate
    error_train = error_cal (data_train, prediction)

    #Output
    metrics = np.array([["entropy:", "%f"%round(entropy,6)], ["error:", "%f"%round(error_train,6)]], dtype=object)

    np.savetxt(outfile, metrics, fmt = "%s")

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    #print("The input file is: %s" % (infile))
    #print("The output file is: %s" % (outfile))

    #Calling the Majority Vote Classifier function
    func(infile, outfile)
