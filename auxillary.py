"""
!TODO
"""

from typing import Callable
import numpy as np

def create_one_hot_vector(data, num_classes=None):
    """Creates one hot vectors"""
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
    """Class of functions"""

    @staticmethod
    def softmax(data: np.ndarray) -> np.ndarray:
        """Softmax function"""
        import scipy  # TODO: Make this our model
        return scipy.special.softmax(data)

    class ActivationFunctions:
        """Activation Functions"""
        @staticmethod
        def sigmoid(data: np.ndarray) -> np.ndarray:
            """Sigmoid"""

            # temp = data - np.max(data) # TODO: Check if required
            temp = data
            exp = np.exp(-temp)
            return 1 / (1 + exp)
            # import scipy as sp
            # return sp.special.expit(x)

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

        if function is Functions.LossFunctions.cross_entropy:
            return Functions.LossFunctions.grad_cross_entropy
        

def map_functions(name: str) -> Callable:

    if name == "sigmoid":
        return Functions.ActivationFunctions.sigmoid
    
    if name == "relu":
        return Functions.ActivationFunctions.relu
    
    if name == "tanh":
        return Functions.ActivationFunctions.tanh
    