"""Dummy classification system.

Dummy solution the COM2004/3004 assignment.

REWRITE THE FUNCTIONS BELOW AND REWRITE THIS DOCSTRING

version: v1.0
"""
from collections import Counter
from typing import List

import numpy as np
import scipy.linalg
import scipy.linalg
from matplotlib import pyplot as plt, cm

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

    covx = np.cov(data, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 20, N - 1))
    v = np.fliplr(v)
    v.shape
    pcatest_data = np.dot((data - np.mean(data)), v)
    # reduced_data = data[:, 0:N_DIMENSIONS]
    return pcatest_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This is your classifier's training stage. You need to learn the model parameters
    from the training vectors and labels that are provided. The parameters of your
    trained model are then stored in the dictionary and returned. Note, the contents
    of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    The dummy implementation stores the labels and the dimensionally reduced training
    vectors. These are what you would need to store if using a non-parametric
    classifier such as a nearest neighbour or k-nearest neighbour classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels
    # e.g. Storing training data labels and feature vectors in the model.
    model = {}
    features_all = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    features_possible = []
    for i in (0,1):
        for j in (2,3):
            for k in (4,5):
                for l in (6,7):
                    for m in (8,9):
                        for n in (10,11):
                            for o in (12,13):
                                for p in (14,15):
                                    for q in (16,17):
                                        for r in (18,19):
                                            if 10 == len(set([i,j,k,l,m,n,o,p,q,r])):
                                                this_features = features_all.copy()
                                                this_features.remove(i)
                                                this_features.remove(j)
                                                this_features.remove(k)
                                                this_features.remove(l)
                                                this_features.remove(m)
                                                this_features.remove(n)
                                                this_features.remove(o)
                                                this_features.remove(p)
                                                this_features.remove(q)
                                                this_features.remove(r)
                                                features_possible.append(this_features)
    alphabet = ['A','B','C','D','E','F','G','H','I','J','K','l','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    features_better = []
    dmax = []
    d = np.zeros(len(features_possible))
    for d_one in alphabet:
        for d_two in alphabet[alphabet.index(d_one)+1:]:
            if d_one != d_two:
                for i in range(len(features_possible)):
                    ndim = 10
                    # compute mean vectors
                    data_one = fvectors_train[labels_train == d_one]
                    data_two = fvectors_train[labels_train == d_two]
                    mu1 = np.mean(data_one[:, features_possible[i]], axis=0)
                    mu2 = np.mean(data_two[:, features_possible[i]], axis=0)

                    # compute distance between means
                    dmu = mu1 - mu2

                    # compute covariance and inverse covariance matrices
                    cov1 = np.cov(data_one[:, features_possible[i]], rowvar=0)
                    cov2 = np.cov(data_two[:, features_possible[i]], rowvar=0)

                    icov1 = np.linalg.inv(cov1)
                    icov2 = np.linalg.inv(cov2)

                    # plug everything into the formula for multivariate gaussian divergence
                    d12 = 0.5 * np.trace(
                        np.dot(icov1, cov2) + np.dot(icov2, cov1) - 2 * np.eye(ndim)
                    ) + 0.5 * np.dot(np.dot(dmu, icov1 + icov2), dmu)
                    d[i] = d12
        dmax.append(d.sum())
        sorted_indexes = np.argsort(-d)
        features_better.append(sorted_indexes[0:10])
        # np.array(features_better)
    list_f = np.array(features_better).flatten()
    d2 = Counter(list_f)
    sorted_x = sorted(d2.items(), key=lambda x: x[1], reverse=True)


    # sss is the test data before
    sss = [(897, 24), (374, 22), (235, 21), (980, 21), (896, 21), (605, 15), (0, 14), (723, 13), (448, 12), (685, 12), (583, 10), (901, 9), (162, 8), (604, 5), (1022, 4), (430, 3), (847, 3), (236, 2), (71, 2), (177, 2), (150, 2), (455, 1), (444, 1), (414, 1), (304, 1), (887, 1), (474, 1), (331, 1), (317, 1), (491, 1), (928, 1), (674, 1), (675, 1), (676, 1), (677, 1), (678, 1), (679, 1), (680, 1), (681, 1), (682, 1), (900, 1), (947, 1), (758, 1), (36, 1), (927, 1), (394, 1), (746, 1), (140, 1), (634, 1), (210, 1), (94, 1), (281, 1), (113, 1), (280, 1), (435, 1), (207, 1)]
    features = [x for x, _ in sorted_x][0:20]
    fvectors_train = np.asarray(fvectors_train)[features]
    label_train = np.asarray(labels_train)[features]
    model["labels_train"] = label_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()

    # model["labels_train"] = labels_train.tolist()
    # fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    # model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Dummy implementation of classify squares.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This is the classification stage. You are passed a list of unlabelled feature
    vectors and the model parameters learn during the training stage. You need to
    classify each feature vector and return a list of labels.

    In the dummy implementation, the label 'E' is returned for every square.

    Args:
        fvectors_train (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    x = np.dot(fvectors_test, fvectors_train)
    modtest = np.sqrt(np.sum(fvectors_test * fvectors_test, axis=1))
    modtrain = np.sqrt(np.sum(fvectors_train * model["fvectors_train"], axis=1))
    dist = x / np.outer(modtest, modtrain.transpose())
    nearest = np.argmax(dist, axis=1)
    mdist = np.max(dist, axis=1)
    label = labels_train[nearest]

    return label


def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """Dummy implementation of find_words.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This function searches for the words in the grid of classified letter labels.
    You are passed the letter labels as a 2-D array and a list of words to search for.
    You need to return a position for each word. The word position should be
    represented as tuples of the form (start_row, start_col, end_row, end_col).

    Note, the model dict that was learnt during training has also been passed to this
    function. Most simple implementations will not need to use this but it is provided
    in case you have ideas that need it.

    In the dummy implementation, the position (0, 0, 1, 1) is returned for every word.

    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """
    return [(0, 0, 1, 1)] * len(words)
