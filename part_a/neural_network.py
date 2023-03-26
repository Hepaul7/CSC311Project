from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        g = self.g(inputs)
        g = torch.sigmoid(g)
        h = self.h(g)
        out = torch.sigmoid(h)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    acc_lst = []  # TODO: Delete later
    epoch_lst = []  # TODO: Delete later
    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + (lamb / 2) * torch.norm(
                model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        acc_lst.append(valid_acc)  # TODO: Delete later
        epoch_lst.append(epoch)  # TODO: Delete later
    return acc_lst, epoch_lst  # TODO: Delete later
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def get_best_epoch(model, lr, train_matrix, zero_train_matrix, valid_data, ub_epochs) -> int:
    """ Find the best epoch for the model. """
    valid_acc, epoch_list = train(model, lr, 0, train_matrix, zero_train_matrix, valid_data,
                                  ub_epochs)

    plt.plot(epoch_list, valid_acc)
    # plt.legend()
    plt.show()

    for i in range(len(valid_acc)):
        if valid_acc[i] == max(valid_acc):
            return epoch_list[i]
    raise ValueError("No best epoch found.")


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    k_list = [10, 50, 100, 200, 500]
    lr_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    lr_list.reverse()

    max_epochs = 100

    best_k = k_list[0]
    best_lr = lr_list[0]
    best_epoch = 0
    max_accuracy = 0
    for k in k_list:
        for lr in lr_list:
            # no regularization term
            train_model = AutoEncoder(train_matrix.shape[1], k)
            curr_best_epoch = get_best_epoch(train_model, lr, train_matrix, zero_train_matrix,
                                             valid_data,
                                             max_epochs)

            test_model = AutoEncoder(train_matrix.shape[1], k)
            train(test_model, lr, 0, train_matrix, zero_train_matrix, valid_data, curr_best_epoch)
            test_acc = evaluate(test_model, zero_train_matrix, test_data)
            if test_acc > max_accuracy:
                max_accuracy = test_acc
                best_lr = lr
                best_k = k
                best_epoch = curr_best_epoch

    print(
        f"Best k: {best_k}, Best lr: {best_lr}, Best num_epoch: {best_epoch}, Best acc: {max_accuracy}")

    # lamb = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    # max_lambda = {}
    # for m in lamb:
    #     model = AutoEncoder(train_matrix.shape[1], max_acc[0])
    #     train(model, max_acc[1], m, train_matrix, zero_train_matrix,
    #           valid_data, max_acc[2])
    #     max_lambda[m] = evaluate(model, zero_train_matrix, valid_data)
    #
    # print("Best k: {}, Best lr: {}, Best num_epoch: {}, Best lamb: {}, Best acc: {}".format(
    #     max(acc, key=acc.get), max_acc))
    #
    # max_hyper = max(acc, key=acc.get)
    # model = AutoEncoder(train_matrix.shape[1], max_hyper[0])
    # epochs = []
    # train_mse = []
    # test_mse = []
    # for epoch in range(0, max_hyper[2]):
    #     train(model, max_hyper[1], max_hyper[3], train_matrix, zero_train_matrix, valid_data,
    #           max_hyper[2])
    #     train_mse.append(evaluate(model, zero_train_matrix, valid_data))
    #     epochs.append(epoch)
    #     test_mse.append(evaluate(model, zero_train_matrix, test_data))
    #
    # plt.plot(epochs, train_mse, label='Train')
    # plt.plot(epochs, test_mse, label='Test')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    # TODO: choose lambda
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
