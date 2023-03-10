'''
CS6910 - Assignment 1
Implementation of Feedforward Neural Network with Backpropagation

@author: cs22m056
'''

from keras.datasets import fashion_mnist
from typing import Callable
import numpy as np
import wandb as wb

wb.init(project="cs6910-assignment-1")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


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
    return data  # TODO


class LayerSizes:
    """LayerSizes"""

    def __init__(self, inp, hidden, output, no_of_hidden_layers):

        self.input_no_neurons = inp
        self.hidden_no_neurons = hidden
        self.output_no_neurons = output
        self.no_of_hidden_layers = no_of_hidden_layers
        self.sizes = np.empty((no_of_hidden_layers + 2, ))

        np.append(self.sizes, input)
        for __ in range(no_of_hidden_layers):
            np.append(self.sizes, hidden)
        np.append(self.sizes, output)


class WeightsAndBiases:
    """Weights and Biases class"""

    def __init__(self, _weights, _bias, _id):

        self._weights = _weights
        self._bias = _bias
        self._id = _id

    def get_weight(self, layer_id):
        """Get Weights"""
        return 0  # TODO

    def get_bias(self, layer_id):
        """Get weights"""
        return 0  # TODO


class Layer:
    """ Basic structure of a Layer """

    def __init__(self, layer_id) -> None:
        self.layer_id = layer_id
        self.weights_incoming = []  # TODO: Is required?
        self.bias_incoming = []  # TODO: Is required
        self.a_pre_activation = []
        self.h_activation = []


class Gradients:
    """Structure holding gradients"""

    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.diff_a = None
        self.diff_h = None
        self.diff_w = None
        self.diff_b = None  # TODO

    @staticmethod
    def sigmoid():
        pass

    @staticmethod
    def tanh():
        pass

    @staticmethod
    def relu():
        pass


def initialize_gradients(sizes):
    """Init Grads"""

    gradients = np.empty((sizes.no_of_hidden_layers, ), dtype=object)

    inp_grad = Gradients(0)

    inp_grad.diff_h = np.zeros((sizes.input_no_of_neurons, 1))
    inp_grad.diff_a = np.zeros((sizes.input_no_of_neurons, 1))

    np.append(gradients, inp_grad)

    for i in range(1, sizes.no_of_hidden_layers + 2):

        grad = Gradients(i)
        grad.diff_a = np.zeros((sizes.sizes[i], 1))
        grad.diff_h = np.zeros((sizes.sizes[i], 1))
        grad.diff_w = np.zeros((sizes.sizes[i], sizes.sizes[i-1]))
        grad.diff_b = np.zeros((sizes.sizes[i], 1))
        np.append(gradients, grad)

    return gradients


def feed_forward_propagation(data, no_of_h_layers, weights_and_biases, activation_function):
    """Q2. Implementation of the feed forward network"""

    layers = np.empty((no_of_h_layers,), dtype=object)

    input_layer = Layer(0)
    input_layer.h_activation = data

    np.append(layers, input_layer)

    for i in range(1, no_of_h_layers + 1):

        layer = Layer(i)

        _w = weights_and_biases.get_weight(i)
        _b = weights_and_biases.get_bias(i)
        h_prev = layers[i - 1].h_activation

        layer.a_pre_activation = np.dot(_w, h_prev) + _b
        layer.h_activation = activation_function(layer.a_pre_activation)

        np.append(layers, layer)

    output_layer = Layer(no_of_h_layers + 1)
    output_layer.a_pre_activation = np.dot(
        weights_and_biases.get_weight(no_of_h_layers + 1),
        layers[no_of_h_layers].h_activation
    ) + weights_and_biases.get_bias(no_of_h_layers + 1)

    output_layer.h_activation = softmax(output_layer.a_pre_activation)
    np.append(layers, output_layer)

    return layers


class ActivationFunctions:
    """AC"""

    # TODO: Make sure all the methods have the same signature

    @staticmethod
    def tanh():
        pass

    @staticmethod
    def sigmoid():
        pass

    @staticmethod
    def relu():
        pass


class LossFunctions:

    # TODO: Make sure all the methods have the same signature

    @staticmethod
    def mean_squared_error():
        pass

    @staticmethod
    def cross_entropy():
        pass


def back_propagation(layers, labels, weights_and_biases: WeightsAndBiases, no_of_h_layers, sizes, grad_activation_function: Callable, loss_function: Callable):
    """Back prop framework"""

    gradients = initialize_gradients(sizes)

    layers[0].a_pre_activation = np.zeros((sizes.input_no_neurons, 1))

    # TODO

    gradients[no_of_h_layers + 1].diff_a = loss_function()  # TODO

    for i in range(no_of_h_layers + 1, 0, -1):  # From output to input

        gradients[i].grad_w = None  # TODO
        gradients[i].grad_b = None  # TODO
        gradients[i].grad_h = None  # TODO
        gradients[i].grad_a = gradients[i-1].grad_h *               \
            grad_activation_function()

    return gradients


if __name__ == "__main__":

    plot_images()
    wb.finish()
