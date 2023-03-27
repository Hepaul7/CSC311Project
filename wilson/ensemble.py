import torch
from sklearn.impute import KNNImputer
from torch.autograd import Variable

from A2.utils import sigmoid
from item_response import irt as model2
from neural_network import train as model3, AutoEncoder

from Project.starter_code.utils import *


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


def model1(matrix, valid_data, k):
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
    return np.array(sparse_matrix_predictions(valid_data, mat))


def get_irt_preds(data, theta, beta):
    """
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
    return np.array(pred)


def get_nn_preds(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    guess = []

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess.append(output[0][valid_data["question_id"][i]].item() >= 0.5)

    return np.array(guess)


def evaluate(final_pred, valid_data):
    """
    :param final_pred: np.array of final predictions (0/1)
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :return: float
    """
    y_true = valid_data['is_correct']
    acc = np.mean(y_true == final_pred)
    return acc


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    train_data = load_train_csv("../data")

    kNN_preds = model1(train_matrix, valid_data, k=21)
    theta, beta, irt_acc, _, _ = model2(train_data, valid_data, 0.01, 100)
    irt_preds = get_irt_preds(valid_data, theta, beta)
    model = AutoEncoder(num_question=train_matrix.shape[1], k=100)
    train_l, valid_l = model3(model, 0.01, 0, train_matrix, zero_train_matrix, valid_data, 25)
    nn_preds = get_nn_preds(model, zero_train_matrix, valid_data)

    final_pred = np.round((kNN_preds + irt_preds + nn_preds) / 3)
    acc = evaluate(final_pred, valid_data)
    print(f"Majority validation accuracy: {acc:.4f}")
    final_test_pred = np.round((model1(train_matrix, test_data, k=21) + get_irt_preds(test_data, theta, beta) + get_nn_preds(model, zero_train_matrix, test_data)) / 3)
    test_acc = evaluate(final_test_pred, test_data)
    print(f"Majority test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
