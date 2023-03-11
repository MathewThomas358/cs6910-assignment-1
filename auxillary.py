import numpy as np

def create_one_hot_vector(y, num_classes=None):
    """Creates one hot vectors"""
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    if num_classes is None:
        num_classes = np.max(y) + 1
    categorical = np.zeros((y.shape[0], num_classes))
    categorical[np.arange(y.shape[0]), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical