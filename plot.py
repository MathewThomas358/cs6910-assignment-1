'''
CS6910 - Assignment 1

Various implementations related to plots and other images

@author: cs22m056
'''

import matplotlib.pyplot as plt
import numpy as np
import wandb as wb

from data import get_data, get_fashion_mnist_original
from network import Optimizers, NeuralNetwork
from auxillary import Functions

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_images():
    """Q1. Function to plot sample images from each class """

    wb.init(project="cs6910-assignment-1")
    x_train, y_train = get_fashion_mnist_original()

    images = []
    labels = []

    for i in range(10):

        index = np.where(y_train == i)
        images.append(x_train[index[0][0]])
        labels.append(class_names[i])

    wb.log({"Samples from each class": [
        wb.Image(img, caption=caption) for img, caption in zip(images, labels)
    ]})
    wb.finish()

def plot_confusion_matrix(
        conf_matrix: np.ndarray,
        classes: list = None,
        cmap=plt.cm.BuPu
):
    """
    Plots the given confusion matrix and logs it.

    Args:
    conf_matrix: np.ndarray - Contains the confusion matrix
    classes: list - A list of names of various classes
    cmap: plt.cm - Colormap to be used. Refer matplotlib colormaps
    """
    wb.init(project="cs6910-assignment-1")

    if classes is None:
        classes = class_names

    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix', fontfamily = "sans-serif", fontsize = "large", fontstretch = "condensed")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.box(False)

    thresh = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, int(conf_matrix[i, j]),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black",
                 fontfamily="fantasy", fontweight = "demibold"
                )

    plt.tight_layout()
    plt.ylabel('True label', fontfamily = "monospace", fontstyle = "oblique")
    plt.xlabel('Predicted label', fontfamily = "monospace", fontstyle = "oblique")
    # plt.show()

    wb.log({"Confusion Matrix": plt})
    wb.finish()

def confusion_matrix(true_labels: list, predicted_labels: list):
    """
    Creates a confusion matrix from the given true and predicted labels.

    Args:
    true_labels: list - List of true labels
    predicted_labels: list - List of predicted labels

    Returns:
    np.ndarray - A square matrix which contains the confusion matrix.
    """

    num_classes = len(np.unique(true_labels))
    confusion_mat = np.zeros((num_classes, num_classes))

    for i, j in zip(true_labels, predicted_labels):
        confusion_mat[int(i) - 1, int(j) - 1] += 1

    return confusion_mat

def plot_conf_matrix(true: list, pred: list):
    """
    Computes the confusion matrix and then generates a plot
    for the confusion matrix and logs it.

    Args:
    true: list - List consisting of true labels.
    pred: list - List consisting of predicted labels.
    """

    conf_matrix = confusion_matrix(true, pred)
    plot_confusion_matrix(conf_matrix, class_names)

def compare_mse_cross_entropy():
    """
    Compares cross entropy loss with mean squared error loss
    and logs the comparison graph.
    """

    train, _, val = get_data()

    opt = Optimizers(
        Functions.ActivationFunctions.tanh,
        Functions.LossFunctions.cross_entropy,
        Functions.softmax,
        10,
        1e-3,
        train[0], train[1],
        16, x_val = val[0],
        y_val = val[1],
        l2_regpara = 5e-3,
        is_sweeping = False,
        beta = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8,
        training_set_size=18000
    )

    nn = NeuralNetwork( #pylint: disable=C0103
        hidden_size = [256],
        is_hidden_layer_size_variable = False,
        no_of_hidden_layers = 3,
        optimizer_function = opt.nadam,
        optimizer_object = opt,
        weight_init = "random"
    )

    cross = nn.train()

    opt = Optimizers(
        Functions.ActivationFunctions.tanh,
        Functions.LossFunctions.squared_loss,
        Functions.softmax,
        10,
        1e-3,
        train[0], train[1],
        16, x_val = val[0],
        y_val = val[1],
        l2_regpara = 5e-3,
        is_sweeping = False,
        beta = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8,
        training_set_size=18000
    )

    nn = NeuralNetwork( #pylint: disable=C0103
        hidden_size = [256],
        is_hidden_layer_size_variable = False,
        no_of_hidden_layers = 3,
        optimizer_function = opt.nadam,
        optimizer_object = opt,
        weight_init = "random"
    )

    mse = nn.train()

    create_loss_graph(cross, mse)

def create_loss_graph(cel: list, mse: list):
    """
    Creates a plot with two line graphs, one denoting cross entropy loss and 
    another for MSE loss.

    Args:
    cel: list - Contains the cross entropy loss
    mse: list - Contains the MSE loss
    """

    wb.init(project="cs6910-assignment-1")

    epochs = range(1, len(cel) + 1)
    plt.plot(epochs, cel, 'r-', label='Cross Entropy Loss')
    plt.plot(epochs, mse, 'b-', label='MSE Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    # plt.show()

    wb.log({"Cross Entropy vs MSE": plt})
    wb.finish()

if __name__ == "__main__":
    compare_mse_cross_entropy()
