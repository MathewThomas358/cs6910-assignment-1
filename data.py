"""#! TODO"""


from keras.datasets import fashion_mnist, mnist

from auxillary import create_one_hot_vector

def get_data(dataset: str = "fashion-mnist"):
    """#! TODO"""

    if dataset == "fashion-mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_val = x_train[54000:]
    y_val = y_train[54000:]
    x_train = x_train[:54000]
    y_train = y_train[:54000]

    x_train = x_train.reshape(x_train.shape[0], 28*28)
    x_val = x_val.reshape(x_val.shape[0], 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_val = x_val / 255.0

    y_train = create_one_hot_vector(y_train)
    y_test = create_one_hot_vector(y_test)
    y_val = create_one_hot_vector(y_val)

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)
    