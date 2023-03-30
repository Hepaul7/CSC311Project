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
    question_meta = load_question_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0  # maybe change to 0.5?
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data, question_meta


class AutoEncoder(nn.Module):
    """
    AutoEncoder class for neural network.
    """

    def __init__(self, num_questions: int, k: int) -> None:
        """ Initialize the AutoEncoder class.
        :param num_questions: number of questions (1774 for this dataset)
        :param k: latent dimension
        """
        super(AutoEncoder, self).__init__()
        # here are the linear functions for the autoencoder
        self.g = nn.Linear(num_questions, k)  # W1 \in R^{n_questions x k}
        self.h = nn.Linear(k, num_questions)  # W2 \in R^{k x n_questions}
        self.k = k

    def get_weight_norm(self) -> float:
        """ Return ||W^1||^2 + ||W^2||^2.
        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs, subject_ids) -> torch.Tensor:
        """ Forward pass of the AutoEncoder.

        :param subject_ids:
        :param inputs: Tensor of user size (1, num_questions)
        :return: Tensor of user size (1, num_questions)
        """
        x = self.g(inputs)
        x = F.sigmoid(x)
        x = self.h(x)
        x = F.sigmoid(x)
        return x


class MultiLayerNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        """ Initialize the AutoEncoder class.
        :param input_size: number of questions (1774 for this dataset)
        :param hidden_sizes: size of each hidden layer (good start is around 25)
        :param output_size: should always be 1 because we are doing binary classification
        """
        super().__init__()

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def get_weight_norm(self) -> float:
        """ Return ||W||^2.
        :return: float
        """
        w_norm = 0
        for layer in self.layers:
            w_norm += torch.norm(layer.weight, 2) ** 2
        return w_norm

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


def compute_ce_loss(train_data, target, output, user_id) -> torch.Tensor:
    """ Compute the cross entropy loss for the given inputs.

    :param user_id:
    :param output:
    :param target:
    :param train_data:
    :return: float
    """
    nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
    target[0][nan_mask] = output[0][nan_mask]

    # Create a boolean mask of valid entries.
    valid_mask = ~torch.isnan(train_data[user_id])

    # print(valid_mask)
    # print(output[0])
    # print(1)
    # print(output[0])
    # print(output[0][valid_mask])

    # Compute the binary cross entropy only for valid entries.
    ce = F.binary_cross_entropy(output[0][valid_mask], target[0][valid_mask],
                                reduction='sum')
    return ce


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
    model.train()  # Tell PyTorch you are training the model.
    # Define optimizers
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    acc_lst, epoch_lst = [], []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # compute cross entropy loss
            ce = compute_ce_loss(train_data, target, output, user_id)
            loss = ce + (lamb / 2) * torch.norm(model.get_weight_norm())
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        acc_lst.append(valid_acc)
        epoch_lst.append(epoch)
    return acc_lst, epoch_lst


def get_best_epoch(model, lr, train_matrix, zero_train_matrix, valid_data, ub_epochs, m):
    """ Find the best epoch for the model. """
    valid_acc, epoch_list = train(model, lr, m, train_matrix, zero_train_matrix, valid_data,
                                  ub_epochs)

    plt.plot(epoch_list, valid_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"lr {lr} k {model.k}")
    plt.show()

    for i in range(len(valid_acc)):
        if valid_acc[i] == max(valid_acc):
            return epoch_list[i], valid_acc[i]
    raise ValueError("No best epoch found.")


def add_subjects():
    """
    For each question that each user has answered. Add all subject vectors together. Then Normalize.
    Here, question data is type dictionary and has {
        "question_id": [],
        "subject_id": []
    }
    :return:
    """
    zero_train_matrix, train_matrix, valid_data, test_data, question_data = load_data()

    user_subjects = {}

    for user_id in range(train_matrix.shape[0]):
        subject_vector = None
        for question_id in range(train_matrix.shape[1]):
            if not np.isnan(train_matrix[user_id][question_id]):
                q_idx = question_data["question_id"].index(question_id)
                question_subject_vector = question_data["subject_id"][q_idx]
                if subject_vector is None:
                    subject_vector = question_subject_vector
                else:
                    for i in range(len(subject_vector)):
                        subject_vector[i] += question_subject_vector[i]
        if user_id not in user_subjects:
            user_subjects[user_id] = subject_vector
        else:
            raise ValueError("Duplicate user_id")
    return user_subjects

def main():
    zero_train_matrix, train_matrix, valid_data, test_data, question_data = load_data()
    user_subjects = add_subjects()
    print(user_subjects.keys())
    print(user_subjects.values())

    # # Set model hyperparameters.
    # # k_list = [10, 50, 100, 200, 500]
    # # lr_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    # k_list = [50]
    # lr_list = [0.01]
    #
    # lr_list.reverse()
    #
    # max_epochs = 50
    #
    # best_k = k_list[0]
    # best_lr = lr_list[0]
    # best_epoch = 0
    # max_accuracy = 0
    # for k in k_list:
    #     for lr in lr_list:
    #         # no regularization term
    #         train_model = AutoEncoder(train_matrix.shape[1], k)
    #         curr_best_epoch, test_acc = get_best_epoch(train_model, lr,
    #                                                    train_matrix, zero_train_matrix,
    #                                                    valid_data, max_epochs, 0)
    #
    #         # test_model = AutoEncoder(train_matrix.shape[1], k)
    #         # train(test_model, lr, 0, train_matrix, zero_train_matrix, valid_data, curr_best_epoch)
    #         # test_acc = evaluate(test_model, zero_train_matrix, test_data)
    #         if test_acc > max_accuracy:
    #             max_accuracy = test_acc
    #             best_lr = lr
    #             best_k = k
    #             best_epoch = curr_best_epoch
    #
    # print(
    #     f"Best k: {best_k}, Best lr: {best_lr}, Best num_epoch: {best_epoch}, Best acc: {max_accuracy}")
    #
    # train_model = AutoEncoder(train_matrix.shape[1], best_k)
    # acc, _ = train(train_model, best_lr, 0, train_matrix, zero_train_matrix, valid_data, best_epoch)
    # valid_acc = evaluate(train_model, zero_train_matrix, valid_data)
    # test_acc = evaluate(train_model, zero_train_matrix, test_data)
    # print(f"Final Validation Acc: {valid_acc}, Test acc: {test_acc}")
    # plt.plot([x for x in range(best_epoch)], acc, label='Train')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    #
    # # lamb = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    # lamb = [0.01]
    # max_accuracy = 0
    # max_lamb = 0
    # for m in lamb:
    #     model = AutoEncoder(train_matrix.shape[1], best_k)
    #     _, test_acc = get_best_epoch(model, best_lr, train_matrix, zero_train_matrix,
    #                                  valid_data, best_epoch, m)
    #     if test_acc > max_accuracy:
    #         max_accuracy = test_acc
    #         max_lamb = m
    #
    # print('Best lambda: ', max_lamb)
    #
    # model = AutoEncoder(train_matrix.shape[1], best_k)
    # train(model, best_lr, max_lamb, train_matrix, zero_train_matrix, valid_data, best_epoch)
    # valid_acc = evaluate(model, zero_train_matrix, valid_data)
    # test_acc = evaluate(model, zero_train_matrix, test_data)
    # print(f"best_lambda: {max_lamb}, Valid acc: {valid_acc}, Test acc: {test_acc}")
    #

if __name__ == "__main__":
    main()