
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
from typing import Callable


#Shuffling the order of input variables for SGD
def shuffle(X, y, epoch):

    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def random_init(shape):

    M, D = shape
    #Initializing seed for consitency and matching the code output with autogrades
    np.random.seed(M*D)
    
    #Values assigned between -0.1 and 0.1
    W = np.random.uniform(low = -0.1, high = 0.1, size = shape)
    return W 


def zero_init(shape):

    return np.zeros(shape = shape)


def softmax(z: np.ndarray) -> np.ndarray:

    soft_sum = np.sum(np.exp(z))
    return np.exp(z)/soft_sum


def cross_entropy(y: int, y_hat: np.ndarray) -> float:

    return np.log(y_hat[y][0])*(-1)


def d_softmax_cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:

    #This gradient is been calculated using differential math taking into account the chain rule
    #Detailed derivation done in HW5 submission, Q2. (b)

    return -y + y_hat


class Sigmoid(object):
    def __init__(self):

        #Create cache to store values for backward pass
        self.cache: dict[str, np.ndarray] = dict()


    def forward(self, x: np.ndarray) -> np.ndarray:

        z = 1/(1+np.exp(-x))
        self.cache["Sigmoid"] = z
        return z
    
    def backward(self, dz: np.ndarray) -> np.ndarray:

        dz_da = self.cache["Sigmoid"]*(1-self.cache["Sigmoid"])
        return dz*dz_da


# This refers to a function type that takes in a tuple of 2 integers (row, col) 
# and returns a numpy array (which should have the specified dimensions).
INIT_FN_TYPE = Callable[[tuple[int, int]], np.ndarray]


class Linear(object):
    def __init__(self, input_size: int, output_size: int, weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        
        # input_size+1 because the bias term will be folded in to the weight vector/matrix
        self.shape = [input_size+1, output_size]

        # Weight matrix initialization (Bias folded in)
        self.w = weight_init_fn(self.shape)

        #Creating a zero matrix array of the same shape to store weight gradients
        self.dw = zero_init(self.shape)

        #Initializing learning rate for SGD
        self.lr = learning_rate

        #Create cache to store values for backward pass
        self.cache: dict[str, np.ndarray] = dict()


    def forward(self, x: np.ndarray) -> np.ndarray:

        #Inserting the bias term at index for the input array
        input_bias = np.insert(x, 0, 1)
        input_bias = input_bias.reshape(len(input_bias), 1)

        #Storing this for use in BackProp
        self.cache["input_bias"] = input_bias

        a = np.dot(np.transpose(self.w), input_bias)

        return a        


    def backward(self, dz: np.ndarray) -> np.ndarray:

        self.dw = np.dot(self.cache["input_bias"], np.transpose(dz))
        return self.dw


    def step(self) -> None:

        self.w -= self.lr*self.dw


class NN(object):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_fn: INIT_FN_TYPE, learning_rate: float):

        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Iinitializing different layers of the NN according to given constraints
        self.linear1 = Linear(self.input_size, self.hidden_size, self.weight_init_fn, learning_rate)
        self.activation = Sigmoid()
        self.linear2 = Linear(self.hidden_size, self.output_size, self.weight_init_fn, learning_rate)


    def forward(self, x: np.ndarray) -> np.ndarray:
        self.a = self.linear1.forward(x)
        self.z = self.activation.forward(self.a)
        self.b = self.linear2.forward(self.z)
        self.y = softmax(self.b)

        return self.y


    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> None:

        self.g_J = 1
        self.g_y = np.divide(-y, y_hat)
        self.g_b = d_softmax_cross_entropy(y, y_hat)
        self.g_beta = self.linear2.backward(self.g_b)
        self.g_z = np.dot(self.linear2.w, self.g_b)
        self.g_a = self.activation.backward(self.g_z[1:]) 
        self.g_alpha = self.linear1.backward(self.g_a)
        self.g_x = self.linear1.backward(self.g_a)


    def step(self):

        self.linear1.step()
        self.linear2.step()


def test(X: np.ndarray, y: np.ndarray, nn: NN) -> tuple[np.ndarray, float]:
    y_pred = np.zeros(len(X))
    
    for idx in range(len(X)):
        y_pred_softmax = nn.forward(X[idx])
        y_pred[idx] = np.argmax(y_pred_softmax)

    #Vectorized form to calculate the difference between two arrays
    error = np.sum(np.array(y_pred) != np.array(y))/len(X)

    return tuple([y_pred, error])


def train(X_tr: np.ndarray, y_tr: np.ndarray, 
          X_test: np.ndarray, y_test: np.ndarray, 
          nn: NN, n_epochs: int) -> tuple[list[float], list[float]]:

    cross_entropy_train = []
    cross_entropy_test = []
   
    #Training the neural network using SGD approach
    for epoch in range(n_epochs):
        
        J_train , J_test = 0, 0
        X_tr_shuff, y_tr_shuff = shuffle(X_tr, y_tr, epoch)
        
        #Looping over each data points for training
        for idx in range(len(X_tr_shuff)):
            y_pred = nn.forward(X_tr_shuff[idx])
            
            y_encoded = np.zeros(shape = y_pred.shape)
            y_encoded[y_tr_shuff[idx]][0] = 1
            
            nn.backward(y_encoded, y_pred)
            nn.step()
        
        #Tabulating the Avg cross entropy after each epoch for the train data
        for idx in range(len(X_tr)):
            y_pred_train = nn.forward(X_tr[idx])
            J_train += cross_entropy(y_tr[idx], y_pred_train)/len(X_tr)
        cross_entropy_train.append(J_train)
        
        #Tabulating the Avg cross entropy after each epoch for the test data
        for idx in range(len(X_test)):
            y_pred_test = nn.forward(X_test[idx])
            J_test += cross_entropy(y_test[idx], y_pred_test)/len(X_test)
        cross_entropy_test.append(J_test)
        
    return cross_entropy_train, cross_entropy_test


if __name__ == "__main__":

    in_train_data = sys.argv[1]
    in_test_data = sys.argv[2] 
    out_tr = sys.argv[3]
    out_te = sys.argv[4]
    out_metrics = sys.argv[5]
    n_epochs = int(sys.argv[6])
    n_hid = int(sys.argv[7])
    init_flag = int(sys.argv[8]) 
    lr = float(sys.argv[9])

    X_train = np.loadtxt(in_train_data, delimiter=',')
    Y_train = X_train[:, 0].astype(int) #Storing the y lable column
    X_train = X_train[:, 1:] #Removing the y label column from the data

    X_test = np.loadtxt(in_test_data, delimiter=',')
    Y_test = X_test[:, 0].astype(int) #Storing the y lable column
    X_test = X_test[:, 1:] #Removing the y label column from the data

    # Defining the labels our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

    #Number of input nodes/featurs
    m_dim = len(X_train[0])

    #Init Flag = 1 ===> Random Initialization of the weight vectors || Init Flag = 2 ===> Zero Initialization of the weight vectors
    if(init_flag == 1):
       nn = NN(m_dim, n_hid, len(labels),  random_init, lr)
    else:
        nn = NN(m_dim, n_hid, len(labels),  zero_init, lr)

    # train the neaural networks
    train_losses, test_losses = train(X_train, Y_train, X_test, Y_test, nn, n_epochs)

    # test the neural networks model (error rate) and making predictions
    train_labels, train_error_rate = test(X_train, Y_train, nn)
    test_labels, test_error_rate = test(X_test, Y_test, nn)

    # Write predicted label and error into file (already implemented for you)
    # Train_losses and test_losses are assumed to be a list of floats are lists of floats
    with open(out_tr, "w") as f:
        for label in train_labels:
            f.write(str(label) + "\n")
    with open(out_te, "w") as f:
        for label in test_labels:
            f.write(str(label) + "\n")
    with open(out_metrics, "w") as f:
        for i in range(len(train_losses)):
            cur_epoch = i + 1
            cur_tr_loss = train_losses[i]
            cur_te_loss = test_losses[i]
            f.write("epoch={} crossentropy(train): {}\n".format(
                cur_epoch, cur_tr_loss))
            f.write("epoch={} crossentropy(validation): {}\n".format(
                cur_epoch, cur_te_loss))
        f.write("error(train): {}\n".format(train_error_rate))
        f.write("error(validation): {}\n".format(test_error_rate))
