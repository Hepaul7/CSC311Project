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
    zero_train_matrix[np.isnan(train_matrix)] = 0  # maybe change to 0.5?
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    """
    AutoEncoder class for neural network.

    """

    def __init__(self, num_questions: int, k: int) -> None:
        """ Initialize the AutoEncoder class.
        :param num_questions:
        :param k:
        """

        super(AutoEncoder, self).__init__()

        # here are the linear functions for the autoencoder
        self.g = nn.Linear(num_questions, k)  # W1 \in R^{n_questions x k}
        self.h = nn.Linear(k, num_questions)  # W2 \in R^{k x n_questions}

    def get_weight_norm(self) -> float:
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs) -> torch.Tensor:
        """ Forward pass of the AutoEncoder.

        :param inputs: Tensor of user size (1, num_questions)
        :return: Tensor of user size (1, num_questions)
        """
        x = self.g(inputs)
        x = F.sigmoid(x)
        x = self.h(x)
        x = F.sigmoid(x)
        return x

    def get_weight(self):
        return self.g.weight


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

            # ce_so_far = 0
            # for i in range(output.shape[1]):
            #     if not np.isnan(train_data[user_id][i]):
            #         ce_so_far += F.binary_cross_entropy(output[0][i], target[0][i])

            # Create a boolean mask of valid entries.
            valid_mask = ~torch.isnan(train_data[user_id])

            # Compute the binary cross entropy only for valid entries.
            ce = F.binary_cross_entropy(output[0][valid_mask], target[0][valid_mask],
                                        reduction='sum')

            loss = ce + (lamb / 2) * torch.norm(
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


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    mod = AutoEncoder(train_matrix.shape[1], 50)
    train(mod, 0.01, 0.01, train_matrix, zero_train_matrix, valid_data, 200)


if __name__ == "__main__":
    main()
