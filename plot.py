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


x_val = x_train[:6000]
y_val = y_train[:6000]
x_train = x_train[:54000]
y_train = y_train[:54000]

x_train = x_train.reshape(x_train.shape[0], 784)
x_val = x_val.reshape(x_val.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

x_train = x_train / 255.0
x_test = x_test / 255.0
x_val  = x_val / 255.0

def plot_images():
    """Q1. Function to plot sample images from each class """

    images = []
    labels = []
    for i in range(10):

        index = np.where(y_train == i)
        images.append(x_train[index[0][0]])
        labels.append(class_names[i])

    wb.log({"Samples from each class": [
        wb.Image(img, caption=caption) for img, caption in zip(images, labels)
    ]})

if __name__ == "__main__":

    plot_images()
    wb.finish()
