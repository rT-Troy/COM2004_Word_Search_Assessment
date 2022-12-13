"""Dummy classification system.

Dummy solution the COM2004/3004 assignment.

REWRITE THE FUNCTIONS BELOW AND REWRITE THIS DOCSTRING

version: v1.0
"""
from typing import List, Union, Tuple

import numpy as np
import scipy.linalg


from utils import utils
from utils.utils import Puzzle

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    Takes the raw feature vectors and reduces them down to the required number of
    dimensions. Note, the `model` dictionary is provided as an argument so that
    you can pass information from the training stage, e.g. if using a dimensionality
    reduction technique that requires training, e.g. PCA.

    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """

    pca_data = np.dot((data - np.array(model["mean"])), np.array(model["matrix"]))      # project the test data to train PCA
    return pca_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """
    Returns:
        dict: a dictionary storing the model data.
    """

    model = {}
    model["labels_train"] = labels_train.tolist()
    covx = np.cov(fvectors_train, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 50, N - 1))
    model["mean"] = np.mean(fvectors_train)
    model["matrix"] = np.fliplr(v).tolist()     # store matrix to model for reduce dimensions function
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    # features = get_features(fvectors_train_reduced,labels_train)
    model["fvectors_train"] = fvectors_train_reduced.tolist()   # reduced dimensions data
    # model["fvectors_train"] = fvectors_train_reduced[:][features].tolist()   # reduced dimensions data
    return model


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """
    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """

    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    x = np.dot(fvectors_test, fvectors_train.transpose())
    modtest = np.sqrt(np.sum(fvectors_test * fvectors_test, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())
    nearest = np.argmax(dist, axis=1)
    label = labels_train[nearest]

    return label


# def get_features(fvectors_train,labels_train):
#     d = []
#     features_all = list(range(0, 50))
#     alphabet = ['A','B','C','D','E','F','G','H','I','J','K','l','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
#     for d_one in alphabet:
#         for d_two in alphabet[alphabet.index(d_one)+1:]:
#             if d_one != d_two and d_one!='Z':
#                 data_one = fvectors_train[labels_train == d_one]
#                 data_two = fvectors_train[labels_train == d_two]
#                 for i in range(len(features_all)):
#                     d = multidivergence(data_one,data_two,features_all[i])
#                 index1 = np.argmax(d)
#                 print(index1)

    # features_all = list(range(0,50))
    # alphabet = ['A','B','C','D','E','F','G','H','I','J','K','l','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    # features_better = []
    # dmax = []
    # d_list = []
    # for d_one in alphabet:
    #     for d_two in alphabet[alphabet.index(d_one)+1:]:
    #         if d_one != d_two and d_one!='Z':
    #             data_one = fvectors_train[labels_train == d_one]
    #             data_two = fvectors_train[labels_train == d_two]
    #             d = multidivergence(data_one,data_two,features_all)
    #
    #         index1 = np.argmax(d)
    #         dmax.append(index1)
    #         d_list.append([d_one,d_two])
    #     sorted_indexes = np.argsort(-d)
    #     features_better.append(sorted_indexes[0:10])
    # list_f = np.array(features_better).flatten()
    # d2 = Counter(list_f)
    # sorted_x = sorted(d2.items(), key=lambda x: x[1], reverse=True)
    # return [x for x, _ in sorted_x][0:20]


# def multidivergence(data_one, data_two, features):
#     ndim = len(features)
#     # compute mean vectors
#
#     mu1 = np.mean(data_one[:, features], axis=0)
#     mu2 = np.mean(data_two[:, features], axis=0)
#
#     # compute distance between means
#     dmu = mu1 - mu2
#
#     # compute covariance and inverse covariance matrices
#     cov1 = np.cov(data_one[:, features], rowvar=0)
#     cov2 = np.cov(data_two[:, features], rowvar=0)
#
#     icov1 = np.linalg.inv(cov1.tolist())
#     icov2 = np.linalg.inv(cov2.tolist())
#
#     # plug everything into the formula for multivariate gaussian divergence
#     d12 = 0.5 * np.trace(
#         np.dot(icov1, cov2) + np.dot(icov2, cov1) - 2 * np.eye(ndim)
#     ) + 0.5 * np.dot(np.dot(dmu, icov1 + icov2), dmu)
#     return d12


def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """
    Returns: list[tuple]: A list of four-element tuples indicating the word positions.
    """

    result_pos = []
    direction = ([0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1])
    for i in range(len(words)):
        correct_rates = []
        possible_result = []
        for row in range(len(labels)):
            for column in range(len(labels[0])):
                label = np.array2string(labels[row][column])[1:2]
                if words[i][:1].upper() == label:
                    desti = destination_axis(row, column, len(labels)-1, len(labels[0])-1, words[i],direction)
                    for k in range(len(desti)):
                        rate, result = words_correct_rate(row, column, desti[k], words[i], labels)
                        correct_rates.append(rate)
                        possible_result.append(result)
        index = correct_rates.index(max(correct_rates))
        result_pos.append(possible_result[index])
    return result_pos
    # return [(0, 0, 1, 1)] * len(words)


def destination_axis(row, column, row_max, column_max, word, direction):
    """

    Args:
        row:
        column:
        row_max:
        column_max:
        word:
        direction:

    Returns: the destination axies and the direction

    """
    destination = []
    minuend = len(word)-1
    left_allow = column-minuend >= 0
    right_allow = column+minuend <= column_max
    up_allow = row-minuend >= 0
    down_allow = row+minuend <= row_max

    if left_allow:
        destination.append(([row, column-minuend], direction[0]))
    if up_allow and left_allow:
        destination.append(([row-minuend,column-minuend],direction[1]))
    if up_allow:
        destination.append(([row-minuend, column], direction[2]))
    if up_allow and right_allow:
        destination.append(([row-minuend, column+minuend], direction[3]))
    if right_allow:
        destination.append(([row, column+minuend], direction[4]))
    if right_allow and down_allow:
        destination.append(([row+minuend, column+minuend], direction[5]))
    if down_allow:
        destination.append(([row+minuend, column], direction[6]))
    if down_allow and left_allow:
        destination.append(([row+minuend, column-minuend], direction[7]))

    return destination


def words_correct_rate(row, column, destination, word, labels):

    correct_num = 0
    for iterate in range(len(word)):
        next_row = row + (destination[1][0])*iterate
        next_column = column + (destination[1][1])*iterate
        letter = word[iterate:iterate + 1].upper()
        label = np.array2string(labels[next_row][next_column])[1:2]
        if label == letter:
            correct_num = correct_num + 1
    correct_rate = correct_num/len(word)
    result = (row, column, destination[0][0], destination[0][1])
    return correct_rate, result
