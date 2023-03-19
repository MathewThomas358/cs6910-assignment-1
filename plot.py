'''
CS6910 - Assignment 1
Implementation of Feedforward Neural Network with Backpropagation

@author: cs22m056
'''

import matplotlib.pyplot as plt
import numpy as np
import wandb as wb

from data import get_data

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_images():
    """Q1. Function to plot sample images from each class """

    wb.init(project="cs6910-assignment-1")
    train, _, _ = get_data()

    x_train = train[0]
    y_train = train[1]

    images = []
    labels = []

    for i in range(10):

        temp = np.zeros_like(y_train[0])
        temp[i] = 1
        index = np.where(y_train == temp)
        images.append(x_train[index[0][0]])
        labels.append(class_names[i])

    wb.log({"Samples from each class": [
        wb.Image(
            np.reshape(img, (-1, 28)), caption=caption
        ) for img, caption in zip(images, labels)
    ]})
    wb.finish()

def plot_confusion_matrix(
        conf_matrix: np.ndarray,
        classes: list = None,
        cmap=plt.cm.BuPu
):
    """
    PLT
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


def confusion_matrix(true_labels, predicted_labels):
    """
    CM
    """

    num_classes = len(np.unique(true_labels))
    confusion_mat = np.zeros((num_classes, num_classes))

    for i, j in zip(true_labels, predicted_labels):
        confusion_mat[int(i) - 1, int(j) - 1] += 1

    return confusion_mat

def plot_conf_matrix(true: list, pred: list):
    """"""

    conf_matrix = confusion_matrix(true, pred)
    plot_confusion_matrix(conf_matrix, class_names)
