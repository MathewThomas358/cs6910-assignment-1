'''
CS6910 - Assignment 1

Performs sweep related operations.

@author: cs22m056
'''

from typing import Any

import numpy as np
import wandb as wb

from network import NeuralNetwork, Optimizers
from auxillary import map_functions, Functions
from data import get_data

CONF = {
    "gamma": 0.9,
    "epsilon": 1e-8,
    "beta": 0.95,
    "beta1": 0.9,
    "beta2": 0.999,
    "dataset": "fashion_mnist"
}

FLAG = False

def init(
        wandb_project:str = "cs6910-assignment-1",
        wandb_entity:str = "cs22m056",
        sweep_conf: dict = None,
        sweep_count: int = 1,
        additional_params: Any = None,
        is_invoked_from_train: bool = False
):
    """
    Initialized a sweep and performs training based on the given configuration.
    """

    global FLAG #pylint: disable=W0603
    FLAG = is_invoked_from_train

    if sweep_conf is None:
        sweep_conf = {

            'method' : 'bayes',
            'metric' : {
            'name' : 'validation_accuracy',
            'goal' : 'maximize'   
            },
            'parameters': {
                'epochs': {
                    'values': [5, 10, 15, 20]
                },
                'is_hidden_layer_size_variable': {
                    'values': [False] #, True]
                },
                'no_of_hidden_layers': {
                    'values': [3, 4, 5]
                },
                'hidden_size_1': {
                    'values': [64, 128, 256, 32]
                },
                'l2_regpara': {
                    'values': [0, 0.5, 0.05, 0.005, 5e-4]
                },
                'learning_rate': {
                    'values': [1e-3, 1e-4, 1e-5] 
                },
                'optimizer': {
                    'values': [ 'nadam', 'momentum', 'nesterov', 'rmsprop', 'adam', 'sgd']
                },
                'batch_size' : {
                    'values':[16, 32, 64, 128, 256]
                },
                'weight_init': {
                    'values': ['random','xavier']
                },
                'activation': {
                    'values': ['sigmoid','tanh','relu']
                }
            }
        }

    if additional_params is not None:
        CONF['beta'] = additional_params.beta
        CONF['beta1'] = additional_params.beta1
        CONF["beta2"] = additional_params.beta2
        CONF['gamma'] = additional_params.momentum
        CONF['epsilon'] = additional_params.epsilon
        CONF['dataset'] = additional_params.dataset

    sweep_id = wb.sweep(sweep_conf, project=wandb_project, entity=wandb_entity)
    wb.agent(sweep_id, sweep, wandb_entity, wandb_project, sweep_count)

def set_hidden_layer(config: dict) -> list:
    """
    Sets the hidden layers according to given config
    
    Args:
    config: dict - Contains the various parameters required to
    train the model.

    """

    hidden_layers = []
    if config.is_hidden_layer_size_variable:

        if config.no_of_hidden_layers == 3:

            hidden_layers.append(config.hidden_size_1)
            hidden_layers.append(config.hidden_size_2)
            hidden_layers.append(config.hidden_size_3)

        if config.no_of_hidden_layers == 4:

            hidden_layers.append(config.hidden_size_1)
            hidden_layers.append(config.hidden_size_2)
            hidden_layers.append(config.hidden_size_3)
            hidden_layers.append(config.hidden_size_4)
        
        if config.no_of_hidden_layers == 4:

            hidden_layers.append(config.hidden_size_1)
            hidden_layers.append(config.hidden_size_2)
            hidden_layers.append(config.hidden_size_3)
            hidden_layers.append(config.hidden_size_4)
            hidden_layers.append(config.hidden_size_5)

    else:
        hidden_layers.append(config.hidden_size_1)

    return hidden_layers

def sweep():
    """
    This function is invoked by wandb inorder to start training as per the given 
    configuration.
    """

    wb.init(config = CONF, resume = "auto")
    config = wb.config
    name = (
        "op_" + str(config.optimizer)[:3] + 
        "_nh_" + str(config.no_of_hidden_layers) + 
        "_bs_" + str(config.batch_size) + 
        "_ac_" + str(config.activation)[:3] + 
        "_hl_" + str(config.hidden_size_1)
    )

    hidden_layers = set_hidden_layer(config)

    wb.run.name = name

    train, test, val = get_data(CONF['dataset'])

    if config.optimizer == "sgd":

        opt = Optimizers(
            map_functions(config.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            config.epochs,
            config.learning_rate,
            train[0], train[1],
            config.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = config.l2_regpara,
            is_sweeping = True
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size = hidden_layers,
            is_hidden_layer_size_variable = config.is_hidden_layer_size_variable,
            no_of_hidden_layers = config.no_of_hidden_layers,
            optimizer_function = opt.gradient_descent,
            optimizer_object = opt,
            weight_init = config.weight_init
        )

    if config.optimizer == "momentum":

        opt = Optimizers(
            map_functions(config.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            config.epochs,
            config.learning_rate,
            train[0], train[1],
            config.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = config.l2_regpara,
            is_sweeping = True,
            gamma = config.gamma
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size = hidden_layers,
            is_hidden_layer_size_variable = config.is_hidden_layer_size_variable,
            no_of_hidden_layers = config.no_of_hidden_layers,
            optimizer_function = opt.momentum_gradient_descent,
            optimizer_object = opt,
            weight_init = config.weight_init
        )

    if config.optimizer == "nesterov" or config.optimizer == "nag":

        opt = Optimizers(
            map_functions(config.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            config.epochs,
            config.learning_rate,
            train[0], train[1],
            config.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = config.l2_regpara,
            is_sweeping = True,
            gamma = config.gamma
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size = hidden_layers,
            is_hidden_layer_size_variable = config.is_hidden_layer_size_variable,
            no_of_hidden_layers = config.no_of_hidden_layers,
            optimizer_function = opt.nesterov_gradient_descent,
            optimizer_object = opt,
            weight_init = config.weight_init
        )

    if config.optimizer == "rmsprop":

        opt = Optimizers(
            map_functions(config.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            config.epochs,
            config.learning_rate,
            train[0], train[1],
            config.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = config.l2_regpara,
            is_sweeping = True,
            epsilon = config.epsilon,
            beta = config.beta
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size = hidden_layers,
            is_hidden_layer_size_variable = config.is_hidden_layer_size_variable,
            no_of_hidden_layers = config.no_of_hidden_layers,
            optimizer_function = opt.rmsprop,
            optimizer_object = opt,
            weight_init = config.weight_init
        )

    if config.optimizer == "adam":

        opt = Optimizers(
            map_functions(config.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            config.epochs,
            config.learning_rate,
            train[0], train[1],
            config.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = config.l2_regpara,
            is_sweeping = True,
            beta = config.beta1,
            beta2 = config.beta2,
            epsilon = config.epsilon
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size = hidden_layers,
            is_hidden_layer_size_variable = config.is_hidden_layer_size_variable,
            no_of_hidden_layers = config.no_of_hidden_layers,
            optimizer_function = opt.adam,
            optimizer_object = opt,
            weight_init = config.weight_init
        )

    if config.optimizer == "nadam":

        opt = Optimizers(
            map_functions(config.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            config.epochs,
            config.learning_rate,
            train[0], train[1],
            config.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = config.l2_regpara,
            is_sweeping = True,
            beta = config.beta1,
            beta2 = config.beta2,
            epsilon = config.epsilon
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size = hidden_layers,
            is_hidden_layer_size_variable = config.is_hidden_layer_size_variable,
            no_of_hidden_layers = config.no_of_hidden_layers,
            optimizer_function = opt.nadam,
            optimizer_object = opt,
            weight_init = config.weight_init
        )

    nn.train()

    if FLAG:

        count = 0
        for i in range(test[0].shape[0]):

            y_pred = nn.predict((test[0])[i, :])
            final = np.zeros_like(y_pred)
            final[np.argmax(y_pred)] = 1

            if ((test[1])[i, :] == final).all():
                count += 1

        print("Accuracy on test", 100 * count/(test[0]).shape[0])
        wb.log({"test_accuracy": 100 * count/(test[0]).shape[0]})

if __name__ == "__main__":
    init(sweep_count=40)
    