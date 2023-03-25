from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer

from Project.starter_code.utils import *


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
    # Transpose the matrix so that we can compute similarity between questions
    transposed_matrix = matrix.T

    # Use KNNImputer to impute missing values
    nbrs = KNNImputer(n_neighbors=k)
    imputed_matrix = nbrs.fit_transform(transposed_matrix)

    # Transpose the imputed matrix back to its original form
    imputed_matrix = imputed_matrix.T

    # Evaluate accuracy on validation data
    acc = sparse_matrix_evaluate(valid_data, imputed_matrix)
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
    k_values = [1, 6, 11, 16, 21, 26]

    val_accuracy = []
    for k in k_values:
        mat = knn_impute_by_user(sparse_matrix, val_data, k)
        val_accuracy.append(mat)

    # Plot validation accuracy as a function of k
    plt.plot(k_values, val_accuracy, '-o')
    plt.title("Validation Accuracy vs. k")
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.show()

    # Choose k with the best performance on validation data
    best_k = k_values[val_accuracy.index(max(val_accuracy))]

    # Compute validation accuracy and test accuracy
    val_acc = max(val_accuracy)
    print("Validation accuracy with k* = %d: %.4f" % (best_k, val_acc))

    test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print("Test accuracy with k* = %d: %.4f" % (best_k, test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
