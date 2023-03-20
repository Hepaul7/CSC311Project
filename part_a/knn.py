from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # transpose the matrix
    matrix_transposed = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix_transposed)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    list_k = [1, 6, 11, 16, 21, 26]
    accuracies = {}
    for k in list_k:
        accuracies[k] = knn_impute_by_user(sparse_matrix, val_data, k)

    # plots the classification rate on the validation set, and etc.     #
    plt.plot(list_k, accuracies.values())
    plt.xlabel("K Value")
    plt.ylabel("Y Value")
    plt.show()

    curr_max = -1
    k_star = 0
    for k in accuracies:
        if accuracies[k] >= curr_max:
            curr_max = accuracies[k]
            k_star = k

    final_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print("k* : ", k_star)
    print("acc k* : ", final_acc)

    # Now computing KNN by item
    accuracies = {}
    for k in list_k:
        accuracies[k] = knn_impute_by_item(sparse_matrix, val_data, k)

    # plots the classification rate on the validation set, and etc.     #
    plt.plot(list_k, accuracies.values())
    plt.xlabel("K Value")
    plt.ylabel("Y Value")
    plt.show()

    curr_max = -1
    k_star = 0
    for k in accuracies:
        if accuracies[k] >= curr_max:
            curr_max = accuracies[k]
            k_star = k

    final_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print("k* : ", k_star)
    print("acc k* : ", final_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
