'''
CS6910 - Assignment 1
Implementation of Feedforward Neural Network with Backpropagation

@author: cs22m056
'''

from typing import Callable

import datetime

from keras.datasets import fashion_mnist

from auxillary import create_one_hot_vector

import numpy as np
import wandb as wb

wb.init(project="cs6910-assignment-1")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

TRAIN_IMAGE_COUNT = 54000

x_val = x_train[:6000]
y_val = y_train[:6000]
x_train = x_train[:54000]
y_train = y_train[:54000]

x_train = x_train.reshape(x_train.shape[0], 784)
x_val  = x_val.reshape(x_val.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

y_train = create_one_hot_vector(y_train)
y_test = create_one_hot_vector(y_test)
y_val = create_one_hot_vector(y_val)


def plot_images():
    """Q1. Function to plot sample images from each class """

    images = []
    labels = []
    for i in range(10):

        index = np.where(y_train == i)
        images.append(x_train[index[0][0]])
        labels.append(class_names[i])

    wb.log({"Samples from each class": [wb.Image(
        img, caption=caption) for img, caption in zip(images, labels)]})


def softmax(data):
    """Softmax function"""
    # data = data - np.max(data)
    return np.exp(data) / np.sum(np.exp(data), axis=0)


class LayerSizes:
    """LayerSizes"""

    def __init__(self, inp: int, hidden: int, output: int, no_of_hidden_layers: int):

        self.input_no_neurons = int(inp)
        self.hidden_no_neurons = int(hidden)
        self.output_no_neurons = int(output)
        self.no_of_hidden_layers = int(no_of_hidden_layers)
        self.sizes = np.array([])

        self.sizes = np.append(self.sizes, inp)
        for __ in range(no_of_hidden_layers):
            self.sizes = np.append(self.sizes, hidden)
        self.sizes = np.append(self.sizes, output)


class WeightsAndBiases:
    """Weights and Biases class"""

    def __init__(self, _id):

        self.weights = None
        self.bias = None
        self._id = _id

    def add(self, gradients, eta: float):
        """Adds two Wbs"""

        self.weights = self.weights + eta * gradients.diff_w
        self.bias = self.bias + eta * gradients.diff_b

    def get_weight(self):
        """W"""
        return self.weights
    
    def get_bias(self):
        """B"""
        return self.bias


def initialize_weights_biases(sizes: LayerSizes):
    """Returns a weights and biases array"""

    # TODO , act_function_str: str, weight_init_type:str
    # TODO Relu cases with Xavier to be written

    size = np.array([])

    weights_and_biases = np.empty(
        (sizes.no_of_hidden_layers + 2, ), dtype=object)
    
    size = np.append(size, sizes.input_no_neurons)
    for __ in range(sizes.no_of_hidden_layers):
        size = np.append(size, sizes.hidden_no_neurons)
    size = np.append(size, sizes.output_no_neurons)

    weights_and_biases[0] = None

    for i in range(1, sizes.no_of_hidden_layers + 2):

        webi = WeightsAndBiases(i)
        # webi.bias = np.zeros((int(size[i]), 1))
        webi.bias = np.random.randn(int(size[i]), 1)
        webi.weights = np.random.randn(int(size[i]), int(size[i-1]))

        weights_and_biases[i] = webi

    return weights_and_biases


class Layer:
    """ Basic structure of a Layer """

    def __init__(self, layer_id) -> None:
        self.layer_id = layer_id
        # self.weights_incoming = []  # TODO: Is required?
        # self.bias_incoming = []  # TODO: Is required
        self.a_pre_activation = []
        self.h_activation = []


class ActivationFunctions:
    """AC"""

    @staticmethod
    def tanh(data):
        """tanh Activation Function"""
        return np.tanh(data)

    @staticmethod
    def sigmoid(data):
        """sigmoid Activation Function"""
        temp = -1 * data
        return 1 / (1 + np.exp(temp))

    @staticmethod
    def relu(data):
        """ReLu Activation Function"""
        # TODO
        pass

    @staticmethod
    def grad_fun(activation_function: Callable) -> Callable:
        """Maps activation functions to it's gradients"""

        if activation_function == ActivationFunctions.sigmoid:
            return Gradients.sigmoid
        
        if activation_function == ActivationFunctions.relu:
            return Gradients.relu
        
        if activation_function == ActivationFunctions.tanh:
            return Gradients.tanh


class Gradients:
    """Structure holding gradients"""

    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.diff_a = None
        self.diff_h = None
        self.diff_w = None
        self.diff_b = None  # TODO Why???????

    @staticmethod
    def sigmoid(data):
        """Gradient of sigmoid function"""
        temp = ActivationFunctions.sigmoid(data)
        return temp * (1 - temp)

    @staticmethod
    def tanh(data):
        """Gradient of tanh function"""
        temp = ActivationFunctions.tanh(data)
        return 1 - temp ** 2

    @staticmethod
    def relu(data):
        """Gradient of ReLu function"""
        # TODO
        pass

    def add(self, grad):
        """Add one gradient to another"""
        self.diff_a = self.diff_a + grad.diff_a
        self.diff_h = self.diff_h + grad.diff_h
        self.diff_w = self.diff_w + grad.diff_w
        self.diff_b = self.diff_b + grad.diff_b


def initialize_gradients(sizes):
    """Init Grads"""

    gradients = np.empty((sizes.no_of_hidden_layers + 2, ), dtype=object)

    inp_grad = Gradients(0)

    inp_grad.diff_h = np.zeros((sizes.input_no_neurons, 1))
    inp_grad.diff_a = np.zeros((sizes.input_no_neurons, 1))

    gradients[0] = inp_grad

    for i in range(1, sizes.no_of_hidden_layers + 2):

        grad = Gradients(i)
        grad.diff_a = np.zeros((int(sizes.sizes[i]), 1))
        grad.diff_h = np.zeros((int(sizes.sizes[i]), 1))
        grad.diff_w = np.zeros((int(sizes.sizes[i]), int(sizes.sizes[i-1])))
        grad.diff_b = np.zeros((int(sizes.sizes[i]), 1))
        gradients[i] = grad

    return gradients


def feed_forward_propagation(data, no_of_h_layers, weights_and_biases, activation_function):
    """Q2. Implementation of the feed forward network"""

    if data.ndim == 1:
      data = np.expand_dims(data, axis=1)

    layers = np.empty((no_of_h_layers + 2,), dtype=object)

    input_layer = Layer(0)
    input_layer.h_activation = data

    layers[0] = input_layer

    for i in range(1, no_of_h_layers + 1):

        layer = Layer(i)

        _w = weights_and_biases[i].get_weight()
        _b = weights_and_biases[i].get_bias()
        h_prev = layers[i - 1].h_activation

        layer.a_pre_activation = np.dot(_w, h_prev) + _b
        layer.h_activation = activation_function(layer.a_pre_activation)

        layers[i] = layer

    output_layer = Layer(no_of_h_layers + 1)
    output_layer.a_pre_activation = np.dot(
        weights_and_biases[no_of_h_layers + 1].get_weight(),
        layers[no_of_h_layers].h_activation
    ) + weights_and_biases[no_of_h_layers + 1].get_bias()

    output_layer.h_activation = softmax(output_layer.a_pre_activation)
    layers[no_of_h_layers + 1] = (output_layer)

    return layers, output_layer.h_activation


class LossFunctions:
    """Loss functions"""

    # TODO: Make sure all the methods have the same signature

    @staticmethod
    def mean_squared_error(x, y):
        """Mean squared error function"""
        return np.mean((y - x)**2)  # TODO: Add regularizer?

    @staticmethod
    def cross_entropy(x, y):
        """Cross entropy function"""
        log_x = np.log(np.array(x).reshape(-1))
        loss = -1 * np.array(y).reshape(-1) * log_x
        return np.sum(loss)  # TODO: Add regularizer?


def back_propagation(layers, true_labels, weights_and_biases, sizes: LayerSizes, grad_activation_function: Callable, loss_function: Callable):
    """Back prop framework"""

    if true_labels.ndim == 1:
      true_labels = np.expand_dims(true_labels, axis=1)

    gradients = initialize_gradients(sizes)

    layers[0].a_pre_activation = np.zeros((sizes.input_no_neurons, 1))

    pred_labels = layers[sizes.no_of_hidden_layers + 1].h_activation

    gradients[sizes.no_of_hidden_layers + 1].diff_a = -1 * (true_labels - pred_labels)  # TODO add case for mse

    for i in range(sizes.no_of_hidden_layers + 1, 0, -1):  # From output to input

        # print(i)
        gradients[i].diff_w = np.dot(gradients[i].diff_a, layers[i-1].h_activation.T)  # TODO Add l2 reg
        gradients[i].diff_b = gradients[i].diff_a
        gradients[i-1].diff_h = np.dot(weights_and_biases[i].weights.T, gradients[i].diff_a)
        gradients[i-1].diff_a = (
            gradients[i-1].diff_h * grad_activation_function(layers[i-1].a_pre_activation)
        )

    return gradients


def sgd(activation_function: Callable, loss_function: Callable, epochs: int, sizes: LayerSizes, eta: float, inti_weight: str):

    weights_and_biases = initialize_weights_biases(sizes)
    # print(sizes.sizes)
    
    for i in range(epochs):

        gradients = initialize_gradients(sizes)
        arr = np.arange(TRAIN_IMAGE_COUNT)
        np.random.shuffle(arr)

        for j in range(TRAIN_IMAGE_COUNT):
            
            print("Epoch: " + str(i) + " Image: " + str(j) + " Time: " + str(datetime.datetime.now().time()))
            x = x_train[arr[j],:]
            # print(y_train)
            # print(y_train[arr[j],:])
            y = y_train[arr[j],:]

            layers, _ = feed_forward_propagation(x, sizes.no_of_hidden_layers, weights_and_biases, activation_function)
            gradients_curr = back_propagation(layers, y, weights_and_biases, sizes, ActivationFunctions.grad_fun(activation_function), loss_function)

            gradients[0].diff_h = gradients[0].diff_h + gradients_curr[0].diff_h
            gradients[0].diff_a = gradients[0].diff_a + gradients_curr[0].diff_a

            for k in range(1, sizes.no_of_hidden_layers + 2):
                gradients[k].add(gradients_curr[k])

        for k in range(1, sizes.no_of_hidden_layers + 2):
            weights_and_biases[k].add(gradients[k], eta)

    return weights_and_biases

def test(a):
    a[1] = 0

if __name__ == "__main__":

    # a = np.array([1,2,3])
    # b = np.array([1,2,3])
    # print(a)

    w_b = sgd(ActivationFunctions.sigmoid, LossFunctions.cross_entropy, 5, LayerSizes(28 * 28, 64, 10, 4), 0.5, None)

    count = 0
    # print(x_val.shape)
    # print(y_val.shape)
    # print(x_val[1,:])

    for a in range(6000):

        _, y_pred = feed_forward_propagation(x_train[a,:], 4, w_b, ActivationFunctions.sigmoid)
        # print((y_val[i,:]))
        # print(y_val[i,:].shape)
        # # y_pred = y_pred[:,]
        # print((y_pred.flatten()))
        # print(y_pred.flatten().shape)
        # break
        print((y_train[a,:] == y_pred.flatten()).all())
        if (y_train[a,:] == y_pred.flatten()).all():       
            count = count + 1

    print((count * 100)/ 6000)
    # plot_images()
    # wb.finish()
