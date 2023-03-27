# TODO: complete this file.
from utils import *
from knn import knn_impute_by_user
from item_response import irt
from neural_network import train, load_data
from matplotlib import pyplot as plt

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def bootstrap_knn(matrix):
    """ Bootstrap the matrix.

    :param matrix: 2D numpy array
    :return: 2D numpy array
    """
    n = len(matrix['user_id'])
    idx = np.random.choice(n, int(np.sqrt(n)), replace=True)
    return matrix[idx, :]


def ensemble_knn():
    """ Compute the ensemble of KNN.
    :return: averaged KNN over 3 bootstrap samples.
    """
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    # test_data = load_public_test_csv("../data")

    bag_1 = bootstrap_knn(sparse_matrix)
    bag_2 = bootstrap_knn(sparse_matrix)
    bag_3 = bootstrap_knn(sparse_matrix)

    k_opt = 11  # from knn.py, note: valid_acc = 0.607112616426757
    knn_1 = knn_impute_by_user(bag_1, val_data, k_opt)
    knn_2 = knn_impute_by_user(bag_2, val_data, k_opt)
    knn_3 = knn_impute_by_user(bag_3, val_data, k_opt)

    return (knn_1 + knn_2 + knn_3) / 3


def bootstrap_irt():
    """
    Bootstrap the data.
    :return: dict of user_id, question_id, is_correct
    """
    train_data = load_train_csv("../data")
    n = len(train_data['user_id'])
    idx = np.random.choice(n, int(np.sqrt(n)), replace=True)
    return {
        'user_id': [train_data['user_id'][i] for i in idx],
        'question_id': [train_data['question_id'][i] for i in idx],
        'is_correct': [train_data['is_correct'][i] for i in idx],
    }


def ensemble_irt():
    """ Compute the ensemble of IRT.
    :return: averaged prediction of 3 IRT models.
    """
    bag_1 = bootstrap_irt()
    bag_2 = bootstrap_irt()
    bag_3 = bootstrap_irt()
    valid_data = load_valid_csv("../data")
    max_param = (0.0001, 50)  # from item_response.py, note: valid_acc = 0.65001411233418

    irt_1 = irt(bag_1, valid_data, max_param[0], max_param[1])[-1][-1]
    irt_2 = irt(bag_2, valid_data, max_param[0], max_param[1])[-1][-1]
    irt_3 = irt(bag_3, valid_data, max_param[0], max_param[1])[-1][-1]
    return (irt_1 + irt_2 + irt_3) / 3


def bootstrap_nn():
    """ Bootstrap the data.
    :return:
    """
    zero_train_matrix, train_matrix, _, _ = load_data()
    n = len(train_matrix['user_id'])
    idx = np.random.choice(n, int(np.sqrt(n)), replace=True)
    return zero_train_matrix[idx, :], train_matrix[idx, :]


def ensemble_nn():
    """ Compute the ensemble of NN.
    :return: averaged prediction of 3 NN models.
    """
    _, _, valid_data, test_data = load_data()

    zero_1, train_1 = bootstrap_nn()
    zero_2, train_2 = bootstrap_nn()
    zero_3, train_3 = bootstrap_nn()

    nn_1, _ = train(zero_1, train_1, valid_data, test_data, 0.0001, 50)
    nn_2, _ = train(zero_2, train_2, valid_data, test_data, 0.0001, 50)
    nn_3, _ = train(zero_3, train_3, valid_data, test_data, 0.0001, 50)

    return (nn_1 + nn_2 + nn_3) / 3


