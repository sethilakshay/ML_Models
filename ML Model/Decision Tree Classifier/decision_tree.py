
"""
Written by Lakshay Sethi

Sharing the code for academic and viewing purposes only!
Prior permission is needed before re-using any part of this code.
Please contact: sethilakshay13@gmail.com for more information.

"""

import sys
import numpy as np

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None
        self.val = None

def entropy_label(data):
    cnt_0 = 0
    cnt_1 = 0
    n = len(data)
    for i in data:
        if (int(i[-1]) == 0):
            cnt_0 += 1
        else:
            cnt_1 += 1
        
    entrpy = -((cnt_0/n)*np.log2(cnt_0/n)) -((cnt_1/n)*np.log2(cnt_1/n))
    
    return entrpy

def entropy_calc(cnt_0, cnt_1, n):
    if (cnt_0 == 0):
        entrpy = -((cnt_1/n)*np.log2(cnt_1/n))
    elif (cnt_1 == 0):
        entrpy = -((cnt_0/n)*np.log2(cnt_0/n))
    else:
        entrpy = -((cnt_0/n)*np.log2(cnt_0/n)) -((cnt_1/n)*np.log2(cnt_1/n)) 

    return entrpy

def info_calc(data, col_split, entrpy_label):
    cnt_attr = {0: 0, 1: 0}
    cnt_lbl_0 = {0: 0, 1: 0}
    cnt_lbl_1 = {0: 0, 1: 0}
    n = len(data)
    
    for i in data:
        #Checking if the value of splitting criterion is 0
        if(int(i[col_split]) == 0):
            cnt_attr[0] += 1
            if(int(i[-1]) == 0):
                cnt_lbl_0[0] += 1
            else:
                cnt_lbl_0[1] += 1
                
        #Checking if the value of splitting criterion is 1        
        else:
            cnt_attr[1] += 1
            if(int(i[-1]) == 0):
                cnt_lbl_1[0] += 1
            else:
                cnt_lbl_1[1] += 1
    entrpy_weighted_0 = (cnt_attr[0]/n)*entropy_calc(cnt_lbl_0[0], cnt_lbl_0[1], cnt_attr[0]) if (cnt_attr[0] != 0) else 0
    entrpy_weighted_1 = (cnt_attr[1]/n)*entropy_calc(cnt_lbl_1[0], cnt_lbl_1[1], cnt_attr[1]) if (cnt_attr[1] != 0) else 0
    info_gain = entrpy_label - entrpy_weighted_0 - entrpy_weighted_1
    
    return info_gain

def majority(data, col_num):
    n = len(data)
    cnt_0 = 0
    cnt_1 = 0
    #Majority Vote Classifier
    for i in data:
        if (int(i[col_num]) == 0):
            cnt_0 += 1
        else:
            cnt_1 += 1
    
    return 0 if cnt_0 > cnt_1 else 1

#Training the Decision Tree
def train_tree(data, max_depth, node):
    #Base Cases
    if(len(data) == 0):
        return
    
    if(max_depth == 0):
        node.val = majority(data, -1)
        node.left = None
        node.right = None
        return
    
    #To check if we have pure labels (Either all 0 or all 1)
    cnt_0 = 0
    cnt_1 = 0
    for i in data:
        if(int(i[-1]) == 0):
            cnt_0 += 1
        else:
            cnt_1 += 1
    
    if((cnt_0 == 0) or (cnt_1 == 0)):
        node.val = 0 if cnt_0 > 0 else 1
        node.left = None
        node.right = None
        return
    
    #Recursion
    entrpy_label = entropy_label(data)  #Calculate the entropy of label
    info_attr = np.zeros(len(data_train[0])-1)
    
    #Compute the information gains of all the attributes
    for i in range(len(data[0])-1):
        info_attr[i] = info_calc(data, i, entrpy_label)
        
    max_info = np.max(info_attr)

    #Only allowing the split to happen if Information Gain greater than 0
    if(max_info == 0):
        node.val = majority(data, -1)
        return
    
    max_info_idx = np.where(info_attr == np.max(info_attr)) #Returns a tuple for indexes wherever a match is found
    max_info_idx = max_info_idx[0][0] #Keeping the first index wherever maximum value was encountered
    
    node.attr = max_info_idx
    
    #Splitting the data in left node (0 label value) and right node (1 label value)
    data_left = []
    data_right = []
    for i in data:
        if(int(i[max_info_idx]) == 0):
            data_left.append(i)
        else:
            data_right.append(i)
    
    #Training the left sub-tree
    node.left = Node()
    train_tree(data_left, max_depth-1, node.left)
    
    #Training the right sub-tree
    node.right = Node()
    train_tree(data_right, max_depth-1, node.right)
    
    return

def predict_tree(row, node):
    if(node.val == None):
        if(int(row[node.attr]) == 0):
            return predict_tree(row, node.left)
        else:
            return predict_tree(row, node.right)
    
    else:
        return node.val

def error_cal (data, res):
    error = 0
    n = len(data)
    for i in range(n):
        if(int(data[i,-1]) != int(res[i])):
            error += 1
    return (error/n)

#Writing Functions for Pretty Print
def data_pretty_cnt(data, col_num):
    n = len(data)
    cnt_0 = 0
    cnt_1 = 0
    for i in data:
        if (int(i[col_num]) == 0):
            cnt_0 += 1
        else:
            cnt_1 += 1
    return cnt_0, cnt_1

def pretty_print(data, tree_depth, node, col_dict):
    if(tree_depth == 0):
        cnt_0, cnt_1 = data_pretty_cnt(data, -1)
        print("[" +str(cnt_0)+" 0/"+str(cnt_1)+" 1]")
        
    tree_depth += 1
    special_char = "| "
    
    if(node.val == None):
        
        data_left = []
        data_right = []
        
        for i in range(len(data)):
            if (int(data[i][node.attr]) == 0):
                data_left.append(data[i])
            else:
                data_right.append(data[i])
            
        cnt_0, cnt_1 = data_pretty_cnt(data_left, -1)
        print(special_char*tree_depth + col_dict[node.attr] + " = 0: " + "[" +str(cnt_0)+" 0/"+str(cnt_1)+" 1]")
            
        pretty_print(data_left, tree_depth, node.left, col_dict)
            
        cnt_0, cnt_1 = data_pretty_cnt(data_right, -1)
        print(special_char*tree_depth + col_dict[node.attr] + " = 1: " + "[" +str(cnt_0)+" 0/"+str(cnt_1)+" 1]")
            
        pretty_print(data_right, tree_depth, node.right, col_dict)
        
        return

#Main Function
if __name__ == '__main__':
    in_train = sys.argv[1]
    in_test = sys.argv[2]
    in_depth = int(sys.argv[3])

    out_train = sys.argv[4]
    out_test = sys.argv[5]
    out_metrics = sys.argv[6]

    data_train = np.genfromtxt(in_train, delimiter="\t", dtype=None, encoding=None)
    data_train_attr = data_train[0]
    data_train = data_train[1:]

    data_test = np.genfromtxt(in_test, delimiter="\t", dtype=None, encoding=None)
    data_test_attr = data_test[0]
    data_test = data_test[1:]

    attr_dict = {}
    for i in range(len(data_train_attr)):
        attr_dict[i] = data_train_attr[i]

    #Checking if Max_depth greater than the number of attributes
    if(len(data_train_attr)-1 < in_depth):
        max_depth = len(data_train_attr)-1
    else:
        max_depth = in_depth

    #Training the tree
    root = Node()
    train_tree(data_train, max_depth, root)

    #Creating two result arrays for storing predicted values
    res_train = np.full((len(data_train), 1), "-1")
    res_test = np.full((len(data_test), 1), "-1")

    #Predicting for Training Data
    for i in range(len(data_train)):
        res_train[i] = "0" if predict_tree(data_train[i], root) == 0 else "1"

    #Predicting for Testing Data
    for i in range(len(data_test)):
        res_test[i] = "0" if predict_tree(data_test[i], root) == 0 else "1"

    #Calculating the error in Training and Testing Data
    error_train = error_cal(data_train, res_train)
    error_test  = error_cal(data_test, res_test)

    metrics = np.array([["error(train):", "%f"%round(error_train,6)], ["error(test):", "%f"%round(error_test,6)]], dtype=object)

    #Saving the output Datasets
    np.savetxt(out_train, res_train, fmt = "%s")
    np.savetxt(out_test, res_test, fmt = "%s")
    np.savetxt(out_metrics, metrics, fmt = "%s")

    #To run pretty print to generate the output simply uncomment the below lines of code
    #data_train: Dataset for which pretty print is called
    #pretty_print(data_train, 0, root, attr_dict)