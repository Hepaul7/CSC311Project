from utils import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch
import matplotlib.pyplot as plt
NUM_QUESTIONS = 1774


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
        question_meta: A dictionary {question_id: list, subject_id: list}
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
        self.g = nn.Linear(num_questions + NUM_SUBJECTS, k)
        self.h = nn.Linear(k, num_questions)
        self.k = k

    def get_weight_norm(self) -> float:
        """ Return ||W^1||^2 + ||W^2||^2.
        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs) -> torch.Tensor:
        """ Forward pass of the AutoEncoder.

        :param inputs: Tensor of user size (1, num_questions + num_subjects)
        :return: Tensor of user size (1, num_questions)
        """

        x = self.g(inputs)
        x = F.sigmoid(x)
        x = self.h(x)
        x = F.sigmoid(x)
        # x = x[:, :1774]  # extract only the first num_questions elements
        # print(x)
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

    :param user_id: int
    :param output: FloatTensor
    :param target: FloatTensor
    :param train_data: FloatTensor
    :return: float
    """
    output = output[:, :NUM_QUESTIONS]
    target = target[:, :NUM_QUESTIONS]

    nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
    target[0][nan_mask] = output[0][nan_mask]

    valid_mask = ~torch.isnan(train_data[user_id])
    ce = F.binary_cross_entropy(output[0][valid_mask], target[0][valid_mask],
                                reduction='sum')
    return ce  # + torch.sum((output - target) ** 2.)


def evaluate(model, train_data, valid_data, subject_vecs) -> float:
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

        inputs = Variable(train_data[u]).unsqueeze(0)  # 1 x 1774
        inputs = concat_input(inputs, subject_vecs, u)

        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, subject_vecs):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param subject_vecs: Dictionary {subject_id: vector}
    :param model: Module (AutoEncoder or MultiLayerNN)
    :param lr: float (learning rate)
    :param lamb: float (regularization parameter)
    :param train_data: 2D FloatTensor (user x question)
    :param zero_train_data: 2D FloatTensor (user x question)
    :param valid_data: Validation Data
    :param num_epoch: int (number of epochs)
    :return: None
    """
    model.train()  # Tell PyTorch you are training the model.
    # Define optimizers
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]
    acc_lst, epoch_lst = [], []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        # print(num_student)
        for user_id in range(num_student):
            # print(user_id)
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)  # 1 x 1774
            inputs = concat_input(inputs, subject_vecs, user_id)  # 1 x 2162
            target = inputs.clone()  # 1 x 2162
            optimizer.zero_grad()
            output = model(inputs)  # 1 x 2162
            # compute cross entropy loss
            ce = compute_ce_loss(train_data, target, output, user_id)
            loss = ce + (lamb / 2) * torch.norm(model.get_weight_norm())
            # loss = F.binary_cross_entropy(output, target, reduction='sum')
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        valid_acc = evaluate(model, zero_train_data, valid_data, subject_vecs)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        acc_lst.append(valid_acc)
        epoch_lst.append(epoch)
    return acc_lst, epoch_lst


def concat_input(inputs, subject_dict, user_id) -> torch.Tensor:
    """ Concatenate the subject vectors to the train_data with matching question_id.
    :param user_id:
    :param inputs: FloatTensor
    :param subject_dict: Dict
    :return: FloatTensor
    """
    if user_id not in subject_dict:
        raise ValueError("User id not found in subject_dict.")

    subject_vec = subject_dict[user_id].unsqueeze(0)
    inputs = torch.cat([inputs, subject_vec], dim=1)
    return inputs


def get_best_epoch(model, lr, train_matrix, zero_train_matrix, valid_data, ub_epochs, m,
                   user_subject):
    """ Find the best epoch for the model.
    :param user_subject: Dictionary of each user and their subject vector
    :param m: float of regularization parameter
    :param ub_epochs: int, upper bound of epochs
    :param model: Module type AutoEncoder
    :param lr: float learning rate
    :param train_matrix: FloatTensor
    :param zero_train_matrix: FloatTensor
    :param valid_data: Dict of validation data
    :return: int of best epoch, float of validation accuracy
    """
    valid_acc, epoch_list = train(model, lr, m, train_matrix, zero_train_matrix, valid_data,
                                  ub_epochs, user_subject)

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
    For each question that each user has answered correctly. Add all subject vectors together.
    Here, question data is type dictionary and has {
        "question_id": [],
        "subject_id": []
    }
    :return: A dictionary that maps each user to all the subjects they have answered correctly.
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
                    subject_vector = [0] * len(question_subject_vector)
                else:
                    for i in range(len(subject_vector)):
                        subject_vector[i] += 1 if question_subject_vector[i] == 1 else 0
        if user_id not in user_subjects:
            user_subjects[user_id] = subject_vector
        else:
            raise ValueError("Duplicate user_id")

    for user_id in range(train_matrix.shape[0]):
        # convert list to tensor
        user_subjects[user_id] = torch.tensor(user_subjects[user_id])
    return user_subjects


def tune_hyperparameters(k_list: list, lr_list: list, max_epochs: int, train_matrix: torch.Tensor,
                         zero_train_matrix: torch.tensor, valid_data: torch.Tensor,
                         user_subjects: dict):
    """
    Tune the hyperparameters for the model. Here, we only tune k, lr and epochs.
    :return:
    """
    best_k = k_list[0]
    best_lr = lr_list[0]
    best_epoch = 0
    max_accuracy = 0
    for k in k_list:
        for lr in lr_list:
            # no regularization term
            train_model = AutoEncoder(train_matrix.shape[1], k)
            curr_best_epoch, test_acc = get_best_epoch(train_model, lr,
                                                       train_matrix, zero_train_matrix,
                                                       valid_data, max_epochs, 0, user_subjects)
            if test_acc > max_accuracy:
                max_accuracy = test_acc
                best_lr = lr
                best_k = k
                best_epoch = curr_best_epoch

    return best_k, best_lr, best_epoch, max_accuracy


def evaluate_best(train_matrix, best_k, best_lr, best_epoch, zero_train_matrix, valid_data,
                  user_subjects, test_data):
    """
    Evaluate the best model.
    :param train_matrix: The train matrix type torch.Tensor
    :param best_k: the Best K we tuned type int
    :param best_lr: The best learning rate we tuned type float
    :param best_epoch: The best epoch we tuned type int
    :param zero_train_matrix: The train matrix with all the NaN replaced with 0 type torch.Tensor
    :param valid_data: The validation data type torch.Tensor
    :param user_subjects: The user subjects, here it is a dictionary that maps each user
    :param test_data: The test data type torch.Tensor
    :return: None, but prints and shows the plot
    """
    train_model = AutoEncoder(train_matrix.shape[1], best_k)
    acc, _ = train(train_model, best_lr, 0, train_matrix, zero_train_matrix, valid_data, best_epoch,
                   user_subjects)
    valid_acc = evaluate(train_model, zero_train_matrix, valid_data, user_subjects)
    test_acc = evaluate(train_model, zero_train_matrix, test_data, user_subjects)
    print(f"Final Validation Acc: {valid_acc}, Test acc: {test_acc}")
    plt.plot([x for x in range(best_epoch)], acc, label='Train')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def tune_lambda(best_k, best_lr, best_epoch, train_matrix, zero_train_matrix, valid_data,
                user_subjects, lamb):
    """
    Tune the regularization term lambda.
    :param lamb: List of lambdas to try.
    :param best_k: Best k value.
    :param best_lr: Best learning rate.
    :param best_epoch: Best epoch.
    :param train_matrix: Training matrix.
    :param zero_train_matrix: Training matrix with all the correct answers.
    :param valid_data: Validation data.
    :param user_subjects: User subjects.
    :return: Best lambda.
    """
    max_accuracy = 0
    max_lamb = 0
    for m in lamb:
        model = AutoEncoder(train_matrix.shape[1], best_k)
        _, test_acc = train(model, best_lr, m, train_matrix, zero_train_matrix, valid_data,
                            best_epoch, user_subjects)
        acc = evaluate(model, zero_train_matrix, valid_data, user_subjects)
        if acc > max_accuracy:
            max_accuracy = acc
            max_lamb = m

    return max_lamb


def evaluate_best_lamb(train_matrix, best_lr, max_lamb, best_k, best_epoch, zero_train_matrix,
                       valid_data, user_subjects, test_data):
    """
    Evaluate the best model with the best lambda.
    :param train_matrix: The train matrix type torch.Tensor
    :param best_lr: The best learning rate we tuned type float
    :param max_lamb: The best lambda we tuned type float
    :param best_k: the Best K we tuned type int
    :param best_epoch: The best epoch we tuned type int
    :param zero_train_matrix: The train matrix with all the NaN replaced with 0 type torch.Tensor
    :param valid_data: The validation data type torch.Tensor
    :param user_subjects: The user subjects, here it is a dictionary that maps each user
    :param test_data: The test data type torch.Tensor
    :return: None, but prints max_lamb and test_acc
    """
    model = AutoEncoder(train_matrix.shape[1], best_k)
    train(model, best_lr, max_lamb, train_matrix, zero_train_matrix, valid_data, best_epoch,
          user_subjects)
    valid_acc = evaluate(model, zero_train_matrix, valid_data, user_subjects)
    test_acc = evaluate(model, zero_train_matrix, test_data, user_subjects)
    print(f"best_lambda: {max_lamb}, Valid acc: {valid_acc}, Test acc: {test_acc}")


def main():
    """
    Main function.
    """
    zero_train_matrix, train_matrix, valid_data, test_data, question_data = load_data()
    user_subjects = add_subjects()

    k_list = [10, 50, 100, 200, 500]
    lr_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    max_epochs = 50
    # k_list = [50]
    # lr_list = [0.01]
    best_k, best_lr, best_epoch, max_accuracy = \
        tune_hyperparameters(k_list, lr_list, max_epochs, train_matrix, zero_train_matrix,
                             valid_data, user_subjects)

    print(
        f"Best k: {best_k}, Best lr: {best_lr},"
        f" "f"Best num_epoch: {best_epoch}, Best acc: {max_accuracy}")

    evaluate_best(train_matrix, best_k, best_lr, best_epoch, zero_train_matrix, valid_data,
                  user_subjects, test_data)

    lamb = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    max_lamb = tune_lambda(best_k, best_lr, best_epoch, train_matrix, zero_train_matrix, valid_data,
                           user_subjects, lamb)

    print('Best lambda: ', max_lamb)
    evaluate_best_lamb(train_matrix, best_lr, max_lamb, best_k, best_epoch, zero_train_matrix,
                       valid_data, user_subjects, test_data)


if __name__ == "__main__":
    main()
