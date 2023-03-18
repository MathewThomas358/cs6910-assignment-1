'''
CS6910 - Assignment 1
Implementation of Feedforward Neural Network with Backpropagation

@author: cs22m056
'''

import numpy as np
import wandb as wb

from data import get_data

wb.init(project="cs6910-assignment-1")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def plot_images():
    """Q1. Function to plot sample images from each class """

    train, _, _ = get_data()

    x_train = train[0]
    y_train = train[1]
    
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
