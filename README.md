# CS6910 Assignment 1

## Author: CS22M056

Link to `wandb` report: https://wandb.ai/cs22m056/cs6910-assignment-1/reports/CS6910-Assignment-1

### Training the model



### Files

The project mainly consists of six files:
1. `train.py` - This script takes command line arguments and starts training the neural network based on the given argument. The options for arguments can be obtained by running `python train.py --help`. This will invoke the `init` function inside `sweep.py`. 
2. `sweep.py` - This script is used to run `wandb` sweeps. This also serves as starting point for training the neural network from `train.py`. If this script is invoked directly, then it starts a sweep with 40 runs, with a given set of hyperparameters. When invoked from `train.py`, it runs a sweep with a single run with the arguments given in command line when `train.py` was invoked.
3. `network.py` - This contains the complete implementation of the neural network and various optimizers. There are three classes inside:
   * `Optimizers` - This class contains the implementation of various optimization algorithms such as gradient descent, stochastic gradient descent, adam, nadam, etc. Variables specific to optimizers have to passed as an argument to the constructor.
   * `Layer` - This class represents a layer of the neural network. Our neural network will be a list which contains multiple layers.
   * `NeuralNetwork` - This class represent the entire neural network. The parameters specific to the neural network, like number of hidden layers, sizes of input, hidden and output layer, optimzer object, and the optimizer function, which will be a member function of the optimizer object. The size of the hidden layer is passed as a list. We can specify the different sizes for each hidden layer by passing it as a list and then setting the `is_hidden_layer_size_variable` as `True`.
4. `plot.py` - This script contains the implementation for generating sample images from each class and for creating the confusion matrix and it's plot.
5. `auxillary.py` - This script contains the various helper functions and classes used by other scripts like `create_one_hot_vector` which create one hot encoded vectors.
6. `data.py` - This script contains the `get_data` function which will take as argument a string, which can be *mnist* or *fashion_mnist* and return 3 tuples, namely training data, test data and validation data. 