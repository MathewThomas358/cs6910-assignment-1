"""
!TODO
"""

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