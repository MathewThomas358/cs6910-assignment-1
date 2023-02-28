'''
CS6910 - Assignment 1
Implementation of Feedforward Neural Network with Backpropagation

@author: cs22m056
'''

from keras.datasets import fashion_mnist

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


class Layer:
    """ Basic structure of a Layer """

    def __init__(self, layer_id, number_of_neurons) -> None:
        self.number_of_neurons = number_of_neurons
        self.layer_id = layer_id
        weights_incoming = []
        bias_incoming = []
        a_pre_activation = []
        h_activation = []


class OutputLayer(Layer):
    """ Basic Structure of an output layer """
    pass


class InputLayer(Layer):
    """ Basic structure of an input layer"""
    pass


def feed_forward_propagation(data, no_of_layers, weights, biases, activation_function):
    """Q2. Implementation of the feed forward network"""

    pre_activation = []  # array of preactivation values obtained from the current iteration

    for i in range(no_of_layers):

        pre_activation_curr = 0
        # TODO: Compute pre

        pre_activation.append(pre_activation_curr)

        activation = activation_function(pre_activation_curr)


if __name__ == "__main__":

    plot_images()
    wb.finish()
