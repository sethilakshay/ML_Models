
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


import numpy as np
import sys


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8', dtype=object)
    return dataset


def split_data_x_y(file):

    x = []
    y = []

    for row in file:
        y.append(int(float(row[0])))
        x.append(list(map(float, row[1:])))     #Converting the list of strings to list of floats

    return x, y


def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(theta, data_X, data_y, num_epoch, learning_rate):
    #Implement this function using vectorization

    for i in range(num_epoch):

        #Using stochastic gradient descent to find the value of theta
        for idx in range(len(data_X)):

            x = [1]     #Adding 1 for the intercept at the start
            x.extend(data_X[idx])
            x = np.array(x)

            y = data_y[idx]

            theta_x = np.dot(theta, x)          #Note that intercept is already included in x and theta at index = 0
            prob_theta_x = sigmoid(theta_x)

            grad = np.dot(x, (prob_theta_x - y))    #Computing the gradient

            theta = np.subtract(theta, learning_rate*grad)
    
    return theta


def predict(theta, data_X):
    #Implement this function using vectorization
    y_pred = []

    for idx in range(len(data_X)):

        x = [1]             #Adding 1 for the intercept at the start
        x.extend(data_X[idx])
        x = np.array(x)

        theta_x = np.dot(theta, x)
        prob_theta_x = sigmoid(theta_x)

        if(prob_theta_x >= 0.5):
            y_pred.append(1)
        else:
            y_pred.append(0)

    return y_pred


def compute_error(y_pred, y):
    #Implement this function using vectorization

    return np.sum((np.array(y_pred) != np.array(y)))/len(y_pred)


if __name__ == '__main__':
    in_feature_train = sys.argv[1]
    in_feature_val = sys.argv[2]
    in_feature_test = sys.argv[3]

    out_train_label = sys.argv[4]
    out_test_label = sys.argv[5]
    out_metrics = sys.argv[6]

    in_epoch = int(sys.argv[7])
    in_learning_rate = float(sys.argv[8])


    #Step 1: Loading the datasets
    data_feature_train = load_tsv_dataset(in_feature_train)
    data_feature_val = load_tsv_dataset(in_feature_val)
    data_feature_test = load_tsv_dataset(in_feature_test)


    #Step 2: Initialzing theta to be 0
    m = len(data_feature_train[0]) - 1      #Subtracting -1 because this contains the label as well
    theta_init = np.zeros(m+1)   #Adding 1 here for the intercept term


    #Step 3: Splitting the data x and y
    data_train_x, data_train_y = split_data_x_y(data_feature_train)
    data_val_x, data_val_y = split_data_x_y(data_feature_val)
    data_test_x, data_test_y = split_data_x_y(data_feature_test)


    #Step 4: Training the model to get final theta vector
    theta = train(theta_init, data_train_x, data_train_y, in_epoch, in_learning_rate)


    #Step 5: Predicting the training and testing values
    train_pred = predict(theta, data_train_x)
    test_pred = predict(theta, data_test_x)


    #Step 6: Computing the error
    error_train = compute_error(train_pred, data_train_y)
    error_test = compute_error(test_pred, data_test_y)

    metrics = np.array([["error(train):", format(round(error_train,6), ".6f")], ["error(test):", format(round(error_test,6), ".6f")]], dtype=object)


    #Step 7: Exporting and outputing the files
   
    np.savetxt(out_train_label, train_pred, fmt = "%s")
    np.savetxt(out_test_label, test_pred, fmt = "%s")
    np.savetxt(out_metrics, metrics, fmt = "%s")