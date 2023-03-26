from matplotlib import pyplot as plt
from utils import *

import numpy as np

from Project.starter_code.utils import *


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    I = np.array(data['user_id'])
    J = np.array(data['question_id'])
    C = np.array(data['is_correct'])
    # Calculate the predicted probabilities for all the answers
    prob_correct = np.exp(theta[I] - beta[J]) / (
                1 + np.exp(theta[I] - beta[J]))

    # Calculate the log-likelihood for all the answers
    log_lklihood = np.sum(
        C * np.log(prob_correct) + (1 - C) * np.log(1 - prob_correct))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    I = np.array(data['user_id'])
    J = np.array(data['question_id'])
    C = np.array(data['is_correct'])

    for i in range(len(theta)):
        theta_grad = np.sum(C[I == i] - sigmoid(theta[i] - beta[J[I == i]]))
        theta[i] += lr * theta_grad

    for j in range(len(beta)):
        beta_grad = np.sum(-C[J == j] + sigmoid(theta[I[J == j]] - beta[j]))
        beta[j] += lr * beta_grad
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.random.randn(len(set(data['user_id'])))
    beta = np.random.randn(len(set(data['question_id'])))

    train_log_lklihood = []
    val_log_lklihood = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        train_log_lklihood.append(neg_lld)
        val_log_lklihood.append(val_neg_lld)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    return theta, beta, val_acc_lst, train_log_lklihood, val_log_lklihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr_values = [0.001, 0.01, 0.1]
    iterations_values = [10, 50, 100]
    best_val_acc = 0.0
    best_lr = None
    best_iterations = None
    tllk = []
    vllk = []
    for lr in lr_values:
        for iterations in iterations_values:
            theta, beta, val_acc, _, _ = irt(train_data, val_data, lr, iterations)
            val_acc = evaluate(val_data, theta, beta)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_lr = lr
                best_iterations = iterations

    # Train the IRT model with the best combination of hyperparameters.
    theta, beta, _, tllk, vllk = irt(train_data, val_data, best_lr, best_iterations)

    # Evaluate the model on the test set.
    test_acc = evaluate(test_data, theta, beta)
    print("Best validation accuracy: {:.2f}%".format(best_val_acc * 100))
    print("Best iteration value: {:d}".format(best_iterations))
    print("Best learning rate: {:.2f}".format(best_lr))
    print("Test accuracy: {:.2f}%".format(test_acc * 100))
    # Plot the training and validation log-likelihoods as a function of iteration for the best hyperparameters.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Training curve
    ax1.plot(tllk, label='train')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-likelihood')
    ax1.set_title('Training Curve')

    # Validation curve
    ax2.plot(vllk, label='validation')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Log-likelihood')
    ax2.set_title('Validation Curve')

    plt.tight_layout()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    # Select three questions
    j1, j2, j3 = 1, 100, 500

    # Define theta range
    theta_range = np.linspace(-4, 4, 100)

    # Calculate the probability of the correct response for each question and theta value
    prob_j1 = sigmoid(theta_range - beta[j1])
    prob_j2 = sigmoid(theta_range - beta[j2])
    prob_j3 = sigmoid(theta_range - beta[j3])

    # Plot the curves
    plt.plot(theta_range, prob_j1, label='Question {}'.format(j1))
    plt.plot(theta_range, prob_j2, label='Question {}'.format(j2))
    plt.plot(theta_range, prob_j3, label='Question {}'.format(j3))

    plt.xlabel('Theta')
    plt.ylabel('P(Correct)')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
