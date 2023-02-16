'''
CS6910 - Assignment 1
Implementation of Gradient Descent with Backpropagation
for Classification

@author: cs22m056
'''

from keras.datasets import fashion_mnist #as data

import numpy as np
import pandas as pd
import wandb as wb

wb.init(project = "cs6910-assignment-1")

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


for i in range(10):

    index = np.where(y_train == i)
    # print()

    wb.log({str(i) : [wb.Image(x_train[index[0][0]])]})

wb.finish()