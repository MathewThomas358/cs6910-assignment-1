"""

Meta

"""

from typing import Callable, Any

import datetime #TEST

from keras.datasets import fashion_mnist

import numpy as np

from auxillary import create_one_hot_vector

def t():
    import time
    time.sleep(100)

RANDOM = "random"
XAVIER = "xavier"

class Layer:
    """Layer"""

    def __init__(self, layer_id:int):
        self.layer_id = layer_id
        self.activation_h: float|None = None
        self.pre_activation_a: float|None = None
        self.weights = None
        self.biases = None

        self.grad_loss_h = None
        self.grad_loss_a = None
        self.grad_loss_w = None
        self.grad_loss_b = None

    def update(self, eta:float) -> None:
        """Update"""
        self.weights = self.weights - eta * self.grad_loss_w
        self.biases = self.biases - eta * self.grad_loss_b

    def reset_grads(self):
        """Reset values"""
        self.grad_loss_h = np.zeros((self.grad_loss_h.shape[0], 1))
        self.grad_loss_a = np.zeros((self.grad_loss_a.shape[0], 1))
        self.grad_loss_w = np.zeros((self.grad_loss_w.shape[0], self.grad_loss_w.shape[1]))
        self.grad_loss_b = np.zeros((self.grad_loss_b.shape[0], 1))

class Functions:
    """Class of functions"""

    @staticmethod
    def softmax(data: np.ndarray) -> np.ndarray:
        """Softmax function"""
        import scipy # TODO: Make this our model
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
        def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
            """Cross entropy loss function"""
            y_pred += 1e-12
            return -np.sum(y_true * np.log(y_pred))

        @staticmethod
        def grad_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
            """Return grad of cross entropy"""
            return y_pred - y_true

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

class Optimizers:
    """O"""

    def __init__(
        self,
        activation_function: Callable = None,
        loss_function: Callable = None,
        output_function: Callable = None,
        epochs: int = None,
        learning_rate: float = None,
        x: np.ndarray = None, # pylint: disable=C0103
        y: np.ndarray = None, # pylint: disable=C0103
        batch_size: int = None
    ):
        assert epochs is not None, ""
        assert activation_function is not None, ""
        assert loss_function is not None, ""
        assert output_function is not None, ""
        assert learning_rate is not None, ""
        assert x is not None, ""
        assert y is not None, ""

        self.activation_function = activation_function
        self.loss_function = loss_function
        self.output_function = output_function
        self.epochs = epochs
        self.eta = learning_rate
        self.x_train = x
        self.y_train = y

        if batch_size is None:
            self.batch_size = x.shape[0]
        else:
            self.batch_size = batch_size

    def gradient_descent(
            self,
            network: list[Layer],
            forward_propagation: Callable,
            back_propagation: Callable,
            is_stochastic: bool = True
        ):
        """SGD and Vanilla GD"""
        
        for j in range(self.epochs):

            training_loss = 0
            
            arr = np.arange(self.x_train.shape[0])
            np.random.shuffle(arr)

            for i in range(self.batch_size):

                x_train_point = self.x_train[arr[i],:]
                y_train_label = self.y_train[arr[i],:]

                network[0].activation_h = np.expand_dims(x_train_point, axis=1)

                _, y_pred = forward_propagation(network, self.activation_function, self.output_function, False)

                training_loss += self.loss_function(y_pred, y_train_label)

                back_propagation(
                    network, y_pred, y_train_label,
                    Functions.get_grad(self.loss_function),
                    Functions.get_grad(self.activation_function)
                )

                if is_stochastic:
                    for i in range(1, network.shape[0]):
                        network[i].update(self.eta)
                    self.__reset_grads_in_network(network)

            if not is_stochastic:
                for i in range(1, network.shape[0]):
                    network[i].update(self.eta)
                    self.__reset_grads_in_network(network)

            print("Epoch", j, "Training Loss:",  training_loss / self.batch_size)

    def momentum_gradient_descent(self):
        pass

    def __reset_grads_in_network(self, network: np.ndarray[Layer, Any]):

        for i in range(1, network.shape[0]):
            network[i].reset_grads()


class NeuralNetwork:
    """Neural Network implementation"""

    def __init__(self,
            input_size: int = 28 * 28,
            hidden_size: list[int] = None,
            output_size: int = 10,
            is_hidden_layer_size_variable = False,
            no_of_hidden_layers: int = 4,
            optimizer_function: Callable = None,
            optimizer_object: Optimizers = None,
            l2_regpara: float = 0,
            weight_init: str = XAVIER
        ):
        """
            
        hidden_size: If the number of neurons in each in layer is the same,
                    then the list will only contain one elements. If the number varies
                    from layer to layer, then provide the entire list.
        """
      
        assert hidden_size is not None, "Hidden size list not provided"
        assert optimizer_function is not None, "Optimizer not provided"
        assert optimizer_object is not None, "Optimzer object not provided"
        self.optimizer: Optimizers = optimizer_function
        self.optimizer_object = optimizer_object
        self.activation_function = optimizer_object.activation_function
        self.loss_function = optimizer_object.loss_function
        self.output_function = optimizer_object.output_function
        self.reg_param = l2_regpara
        self.total_layers = no_of_hidden_layers + 2
        self.weight_init = weight_init
        self.testing = True # TODO Remove

        if is_hidden_layer_size_variable:
            self.sizes = [input_size] + hidden_size + [output_size]
        else:
            self.sizes = [input_size]
            for _ in range(no_of_hidden_layers):
                self.sizes = self.sizes + [hidden_size[0]]
            self.sizes = self.sizes + [output_size]

        self.network = np.empty((self.total_layers, ), dtype=Layer)

        self.network[0] = Layer(0)
        self.network[0].pre_activation_a = np.zeros((self.sizes[0], 1), dtype=np.float64)

        for i in range(1, self.total_layers):

            layer = Layer(i)
            self.__weight_init(layer, i)
            layer.grad_loss_a = np.zeros((self.sizes[i], 1), dtype=np.float64)
            layer.grad_loss_h = np.zeros((self.sizes[i], 1), dtype=np.float64)
            layer.grad_loss_b = np.zeros((self.sizes[i], 1), dtype=np.float64)
            layer.grad_loss_w = np.zeros((self.sizes[i], self.sizes[i-1]), dtype=np.float64)

            self.network[i] = layer

    def __weight_init(self, layer: Layer, i: int):

        if self.weight_init == RANDOM:

            if self.optimizer_object.activation_function is Functions.ActivationFunctions.relu:
                factor = np.sqrt(2 / (self.sizes[i] + self.sizes[i-1]))
                layer.weights = factor * np.random.randn(self.sizes[i], self.sizes[i-1])
                layer.biases = np.zeros((self.sizes[i], 1))
            else:
                layer.biases = np.random.rand(int(self.sizes[i]), 1)
                layer.weights = np.random.rand(int(self.sizes[i]), int(self.sizes[i-1]))

        if self.weight_init == XAVIER:

            if self.optimizer_object.activation_function is Functions.ActivationFunctions.relu:
                factor = np.sqrt(2 / self.sizes[i-1])
                layer.weights = factor * np.random.rand(int(self.sizes[i]), int(self.sizes[i-1]))
                layer.biases = factor * np.random.rand(int(self.sizes[i]), 1)
            else:
                factor = 2 / np.sqrt(self.sizes[i-1])
                layer.weights = factor * (-0.5 + np.random.rand(int(self.sizes[i]), int(self.sizes[i-1])))
                layer.biases = factor * np.random.rand(int(self.sizes[i]), 1)

    def feed_forward_propagation(self,
        layers: list[Layer],
        activation_function: Callable,
        output_activation_function: Callable,
        test_flag = False
    ):
        """FFWD"""
        
        # if test_flag: print("0", layers[0].activation_h.flatten())

        for i in range(1, self.total_layers - 1):
            
            layers[i].pre_activation_a = (
                layers[i].biases +
                np.dot(layers[i].weights, layers[i-1].activation_h)
            )

            # if self.testing and test_flag:
            #     self.testing = False
            #     print("W", np.max(layers[i].weights), layers[i].weights.flatten())
            #     print("B", np.max(layers[i].biases), layers[i].biases.flatten())
            #     print("D", np.max(layers[i-1].activation_h), layers[i-1].activation_h.flatten())

            # if test_flag: print("P", i, layers[i].pre_activation_a.shape, layers[i].pre_activation_a.flatten())

            layers[i].activation_h = activation_function(
                layers[i].pre_activation_a
            )
            # if test_flag: print("A", i, layers[i].activation_h.shape, layers[i].activation_h.flatten())

        layers[self.total_layers - 1].pre_activation_a = (
                layers[self.total_layers - 1].biases +
                np.dot(
                    layers[self.total_layers - 1].weights,
                    layers[self.total_layers - 2].activation_h
                )
        )
        # if test_flag: print("PL", self.total_layers - 1, layers[self.total_layers - 1].pre_activation_a.flatten())

        y_pred = output_activation_function(layers[self.total_layers - 1].pre_activation_a)
        # if test_flag: print("AL", self.total_layers - 1, y_pred.flatten())

        return layers, y_pred

    def back_propagation(
        self,
        layers: list[Layer],
        y_pred: np.ndarray,
        y_true: np.ndarray,
        gradient_loss_function: Callable,
        gradient_activation_function: Callable
    ):
        """BP"""
        y_true = np.expand_dims(y_true, axis=1)
        layers[self.total_layers - 1].grad_loss_a = gradient_loss_function(y_pred, y_true)

        for i in range(self.total_layers - 1, 0, -1):

            layers[i].grad_loss_w += np.dot(
                layers[i].grad_loss_a,
                layers[i-1].activation_h.T
            ) + self.reg_param * layers[i].weights # TODO: Add regparam factor in computing the loss
            
            assert layers[i].grad_loss_w is not None, i

            layers[i].grad_loss_b += layers[i].grad_loss_a

            layers[i-1].grad_loss_h = np.dot(layers[i].weights.T, layers[i].grad_loss_a)
            layers[i-1].grad_loss_a = (
                layers[i-1].grad_loss_h *
                gradient_activation_function(
                    layers[i-1].pre_activation_a
                )
            )
        
        return layers
    
    def print(self):
        """Print"""

        for i in range(1, self.network.shape[0]):
            print("W", i, self.network[i].weights.flatten())
            print("B", i, self.network[i].biases.flatten())

    def train(self):
        """Train"""
        return self.optimizer(self.network, self.feed_forward_propagation, self.back_propagation)

    def predict(self, x_test):
        """Predict the class of the given x"""
        self.network[0].activation_h = np.expand_dims(x_test, axis=1)
        _, y_pred = self.feed_forward_propagation(self.network, self.activation_function, self.output_function, False)
        assert 1+1e-9 > np.sum(y_pred) > 1-1e-9
        return y_pred.flatten()


def main2():
    ''' Meow'''
    
    # size_of_matrix = 3
    # num_train_data = 1024 * 1024
    # num_test_data = 1024
    # x_train = np.random.random((num_train_data, size_of_matrix, size_of_matrix))
    # x_test = np.random.random((num_test_data, size_of_matrix, size_of_matrix))
    # y_train = np.sum(np.argmax(x_train, axis=-1), axis = 1)
    # y_test = np.sum(np.argmax(x_test, axis=-1), axis = 1)

    # x_train = x_train.reshape(x_train.shape[0], size_of_matrix * size_of_matrix)
    # x_test = x_test.reshape(x_test.shape[0], size_of_matrix * size_of_matrix)
    
    size_of_matrix = 9
    num_train_data = 1024 * 1024
    num_test_data = 1024
    x_train = np.random.random((num_train_data, size_of_matrix))
    x_test = np.random.random((num_test_data, size_of_matrix))
    y_train = np.argmax(x_train, axis=1)
    y_test = np.argmax(x_test, axis=1)

    x_train = x_train.reshape(x_train.shape[0], size_of_matrix)
    x_test = x_test.reshape(x_test.shape[0], size_of_matrix)

    y_train = create_one_hot_vector(y_train)
    y_test = create_one_hot_vector(y_test)

    opti = Optimizers(
        Functions.ActivationFunctions.sigmoid,
        Functions.LossFunctions.cross_entropy,
        Functions.softmax,
        1200, 5e-3, x_train, y_train,
        4096
    )
    
    nn = NeuralNetwork(9, [5], 9, False, 1, opti.gradient_descent, opti) #pylint: disable=C0103
    nn.train()

    count = 0
    temp = 10
    for i in range(x_test.shape[0]):

        y_pred = nn.predict(x_test[i,:])
        final = np.zeros_like(y_pred)
        final[np.argmax(y_pred)] = 1
        if temp >0:
            tq2=" ".join([f"{q:1.3f}" for q in x_test[i,:]])
            tq= " ".join([f"{q:1.3f}" for q in y_pred])
            # print(f"{tq2} --{temp}: {tq} - {final} - {y_test[i,:]}")
            print(f"{tq2}<>{tq}-{y_test[i,:]}")
            temp -=1
        # print("True", y_val[i,:])
        if (y_test[i,:] == final).all():
            count += 1

    print(100 * count/x_test.shape[0])

def main():
    """Main"""

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_val = x_train[:6000]
    y_val = y_train[:6000]
    x_train = x_train[:54000]
    y_train = y_train[:54000]

    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_val = x_val.reshape(x_val.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_val  = x_val / 255.0

    y_train = create_one_hot_vector(y_train)
    y_test = create_one_hot_vector(y_test)
    y_val = create_one_hot_vector(y_val)

    optimizer = Optimizers(
        Functions.ActivationFunctions.tanh,
        Functions.LossFunctions.cross_entropy,
        Functions.softmax,
        25, 5e-3, x_train, y_train, 1024
    )

    nn = NeuralNetwork(                                     #pylint: disable=C0103
        28 * 28, [64], 10,
        False, 4,
        optimizer.gradient_descent,
        optimizer
    )

    nn.train()

    count = 0
    temp = 10

    for i in range(x_val.shape[0]):

        y_pred = nn.predict(x_val[i,:])
        final = np.zeros_like(y_pred)
        final[np.argmax(y_pred)] = 1
        if temp > 0:
            # tq2=" ".join([f"{q:1.3f}" for q in x_val[i,:]])
            t_q= " ".join([f"{q:1.3f}" for q in y_pred])
            print(f"{temp}: {t_q} - {y_val[i,:]}")
            temp -= 1
            
        # print("Pred", y_pred)
        if (y_val[i,:] == final).all():
            count += 1

    print(100 * count/x_val.shape[0])

if __name__ == "__main__":
    main()
