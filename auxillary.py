"""
CS6910 - Assignment 1

Helper script which contains some helpful functions and classes

@author: cs22m056
"""

from typing import Callable

import numpy as np
import wandb as wb

def create_one_hot_vector(data, num_classes=None) -> np.ndarray:
    """Creates one hot vectors
    
    Args:
    data: list - The list which contains the labels
    num_classes - Number of classes

    Returns:
    np.ndarray - A 2D array which contains data.length rows, where row i contains
    a one-hot vector corresponding to data[i]
    """

    data = np.array(data, dtype='int')
    input_shape = data.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    if num_classes is None:
        num_classes = np.max(data) + 1
    categorical = np.zeros((data.shape[0], num_classes))
    categorical[np.arange(data.shape[0]), data] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)

    return categorical

class Functions:
    """
    Implementations of commonly used functions and their gradients.
    """

    @staticmethod
    def softmax(data: np.ndarray) -> np.ndarray:
        """Softmax function"""
        data = data - np.max(data)
        exp = np.exp(data)
        return exp / np.sum(exp, axis = 0)

    class ActivationFunctions:
        """Activation Functions"""
        @staticmethod
        def sigmoid(data: np.ndarray) -> np.ndarray:
            """Sigmoid"""
            temp = data
            exp = np.exp(-temp)
            return 1 / (1 + exp)

        @staticmethod
        def relu(data: np.ndarray) -> np.ndarray:
            """REctified Linear Unit function"""
            return np.maximum(0, data)

        @staticmethod
        def tanh(data: np.ndarray) -> np.ndarray:
            """tanh function"""
            return np.tanh(data)

    class LossFunctions:
        """Loss Functions"""
        @staticmethod
        def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray, norm: float = 0, lamda: float = 0) -> float:
            """Cross entropy loss function
                P.S. I know it's lambda and not lamda:)
            """
            y_pred += 1e-12
            return -np.sum(y_true * np.log(y_pred)) + (lamda / 2) * norm

        @staticmethod
        def grad_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            """Return grad of cross entropy"""
            return y_pred - y_true

        @staticmethod
        def squared_loss(y_pred: np.ndarray, y_true: np.ndarray, norm: float = 0, lamda: float = 0) -> float:
            """SQ"""
            return np.mean((y_pred - y_true) * (y_pred - y_true)) + (lamda / 2) * norm

    class GradientActivationFunctions:
        """Gradient of Activation Functions"""
        @staticmethod
        def grad_sigmoid(data: np.ndarray) -> np.ndarray:
            """Grad sigmoid"""
            temp = Functions.ActivationFunctions.sigmoid(data)
            return temp * (1 - temp)

        @staticmethod
        def grad_relu(data: np.ndarray) -> np.ndarray:
            """Gradient of ReLU functions"""
            grad = np.zeros_like(data)
            grad[data >= 0] = 1
            return grad

        @staticmethod
        def grad_tanh(data: np.ndarray) -> np.ndarray:
            """Gradient of tanh function"""
            return 1 - np.tanh(data) ** 2

    @staticmethod
    def get_grad(function):
        """Return grad of function"""

        if function is Functions.ActivationFunctions.sigmoid:
            return Functions.GradientActivationFunctions.grad_sigmoid

        if function is Functions.ActivationFunctions.tanh:
            return Functions.GradientActivationFunctions.grad_tanh

        if function is Functions.ActivationFunctions.relu:
            return Functions.GradientActivationFunctions.grad_relu

        if function is Functions.LossFunctions.cross_entropy or \
            function is Functions.LossFunctions.squared_loss:
            return Functions.LossFunctions.grad_cross_entropy

def map_functions(name: str) -> Callable:
    """
    Return the function related to the given name

    Args:
    name: str - Name of the function

    Returns: 
    Callable: Function corresponding to the name
    """

    if name == "sigmoid":
        return Functions.ActivationFunctions.sigmoid

    if name == "relu":
        return Functions.ActivationFunctions.relu

    if name == "tanh":
        return Functions.ActivationFunctions.tanh

    if name == "mean_squared_error":
        return Functions.LossFunctions.squared_loss

    if name == "cross_entropy":
        return Functions.LossFunctions.cross_entropy

def evaluate_metrics_and_log(
    training_loss: float,
    training_accuracy: float,
    x_val: np.ndarray,
    y_val: np.ndarray,
    network: np.ndarray,
    forward_propagation: Callable,
    activation_function: Callable,
    output_function: Callable,
    loss_function: Callable,
    lamda: float,
    norm: float
):
    """Used to evaluate the training and validation accuracies and losses"""
    #! TODO: Move to auxillary

    validation_hits = 0
    validation_loss = 0

    for i in range(x_val.shape[0]):

        x_val_point = x_val[i, :]
        y_val_label = y_val[i, :]

        network[0].activation_h = np.expand_dims(x_val_point, axis=1)

        _, y_pred = forward_propagation(
            network, activation_function, output_function)

        if np.argmax(y_pred.flatten()) == np.argmax(y_val_label):
            validation_hits += 1

        validation_loss += loss_function(y_pred,
                                         y_val_label, norm, lamda)  # ! TODO

    validation_accuracy = validation_hits / x_val.shape[0]
    validation_loss = validation_loss / x_val.shape[0]

    metrics = {
        "training_accuracy": float(training_accuracy),
        "training_loss": float(training_loss),
        "validation_accuracy": float(validation_accuracy),
        "validation_loss": float(validation_loss)
    }

    wb.log(metrics)
    print(metrics)
