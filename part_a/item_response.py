from utils import *

import numpy as np


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
    log_lklihood = 0
    for i, u in enumerate(data['user_id']):
        log_lklihood += data['is_correct'][i] * \
                        (theta[u] - beta[data['question_id'][i]]) - \
                        np.log(1 + np.exp(theta[u] - beta[data['question_id'][i]]))
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
    user_id = np.array(data['user_id'])
    question_id = np.array(data['question_id'])
    is_correct = np.array(data['is_correct'])
    for i in range(len(theta)):
        theta[i] += lr * np.sum(is_correct[user_id == i]
                                - sigmoid(theta[i] - beta[question_id[user_id == i]]))
    for i in range(len(beta)):
        beta[i] += lr * np.sum(is_correct[question_id == i] -
                               sigmoid(theta[user_id[question_id == i]] - beta[i]))

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
    # TODO: Initialize theta and beta.
    num_users = len(set(data["user_id"]))
    num_questions = len(set(data["question_id"]))
    theta = np.random.randn(num_users) * 0.01
    beta = np.random.randn(num_questions) * 0.01
    # theta = np.ones(num_users)
    # beta = np.ones(num_questions)
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


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
    lrs = [x / 10000 for x in range(1, 3)]
    iterations = [50]

    acc = {}
    for lr in lrs:
        for iteration in iterations:
            acc_iter = irt(train_data, val_data, lr, iteration)[-1][-1]
            acc[(lr, iteration)] = acc_iter
            print(acc)
    max_acc_so_far = 0
    max_params = (lrs[0], iterations[0])
    for i in acc:
        if acc[i] > max_acc_so_far:
            max_acc_so_far = acc[i]
            max_params = i[0], i[1]
    print(max_params)

    # students = {}
    # for i in range(len(train_data['user_id'])):
    #     if train_data['user_id'][i] not in students:
    #         students[train_data['user_id'][i]] = 1
    #     else:
    #         students[train_data['user_id'][i]] += 1
    # print(students)
    # print(len(students))
    #
    # questions = {}
    # for i in range(len(train_data['question_id'])):
    #     if train_data['question_id'][i] not in questions:
    #         questions[train_data['question_id'][i]] = 1
    #     else:
    #         questions[train_data['question_id'][i]] += 1
    # print(questions)
    # print(len(questions))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
