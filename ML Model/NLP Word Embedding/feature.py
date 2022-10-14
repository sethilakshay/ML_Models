
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


import csv
import numpy as np
import sys

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt


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


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map


def word_trim(data, word2vec):
    """
    Trims the sentences in file by checking if the words are present 
    in the feature embedding vector

    Parameters:
        data (list): Contains the list of sentences (itself a list) input by the user
        word2vec (dict): Dictionary of words as keys along with their embeddings as values

    Returns:
        A list of list indexed in the same order as the data, where the first element 
        in this list is label and the second element is the list of trimmed words
        found in the word2vec
    """
    trimmed_sentences = []

    #Looping over the input file
    for sentence in data:

        label = float(sentence[0])        #Extracting label which is stored at 0th index
        sentence_words = sentence[1].split(" ")     ##Extracting words from sentence by splitting the 1st index
        trimmed_sentence_words = []

        for j in sentence_words:
            #Checking if the word is present in the feature vector
            if j in word2vec:
                trimmed_sentence_words.append(j)
        
        trimmed_sentences.append([label, trimmed_sentence_words])

    return trimmed_sentences


def sentence_feature(trimmed_data, word2vec):
    output_list = []

    #Getting the dimensions of the feature vector
    m = len(word2vec[list(word2vec.keys())[0]])

    for trimmed_sentence in trimmed_data:
        output_str = format(trimmed_sentence[0], '.6f') + "\t"      #Adding the label at the start of the string

        for dimension in range(m):
            
            feature_score = 0   
            for word in trimmed_sentence[1]:       #Computing the score here 

                feature_score += word2vec[word][dimension]/len(trimmed_sentence[1])

            output_str += format(round(feature_score, 6), '.6f') + "\t"

        output_list.append(output_str.rstrip())

    return output_list


#Main function
if __name__ == '__main__':
    in_train = sys.argv[1]
    in_val = sys.argv[2]
    in_test = sys.argv[3]

    word2vec_embeddings = sys.argv[4]

    out_train = sys.argv[5]
    out_val = sys.argv[6]
    out_test = sys.argv[7]

    #Calling the load_tsv_dataset function
    data_train = load_tsv_dataset(in_train)
    data_val = load_tsv_dataset(in_val)
    data_test = load_tsv_dataset(in_test)

    #Loading the Feature vector Word Embeddings
    feature_embeddings_map = load_feature_dictionary(word2vec_embeddings)

    #Step 1: Checking if the words of a sentence are present in feature embeddings
    trimmed_train = word_trim(data_train, feature_embeddings_map)
    trimmed_val = word_trim(data_val, feature_embeddings_map)
    trimmed_test = word_trim(data_test, feature_embeddings_map)

    #Step 2: Generating the feature score and desired output
    feature_train = sentence_feature(trimmed_train, feature_embeddings_map)
    feature_val = sentence_feature(trimmed_val, feature_embeddings_map)
    feature_test = sentence_feature(trimmed_test, feature_embeddings_map)

    #Step 3: Saving the desired output file
    np.savetxt(out_train, feature_train, fmt = "%s")
    np.savetxt(out_val, feature_val, fmt = "%s")
    np.savetxt(out_test, feature_test, fmt = "%s")