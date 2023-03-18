"""
    Sweep
"""

import wandb as wb

from network import NeuralNetwork, Optimizers
from auxillary import map_functions, Functions
from data import get_data

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
            'values': [True, False]
        },
        'no_of_hidden_layers': {
            'values': [3, 4, 5]
        },
        'hidden_size_1': {
            'values': [32, 64, 128, 256]
        },
        'hidden_size_2': {
            'values': [32, 64, 128, 256]
        },
        'hidden_size_3': {
            'values': [32, 64, 128, 256]
        },
        'hidden_size_4': {
            'values': [32, 64, 128, 256]
        },
        'hidden_size_5': {
            'values': [32, 64, 128, 256]
        },
        'l2_regpara': {
            'values': [0, 0.5, 0.05, 0.005]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4, 1e-5] 
        },
        'optimizer': {
            'values': [ 'sgd' , 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
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

sweep_id = wb.sweep(sweep_conf, project="cs6910-assignment-1", entity="cs22m056")

def set_hidden_layer(config: dict) -> list:

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
    """#! TODO"""

    conf = {} #? Is this supposed to be empty?
    wb.init(config = conf, resume = "auto")
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

    train, _, val = get_data()

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
            ,training_set_size = 2048 #TEST
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
            gamma = 0.9
            ,training_set_size = 2048 #TEST
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size = hidden_layers,
            is_hidden_layer_size_variable = config.is_hidden_layer_size_variable,
            no_of_hidden_layers = config.no_of_hidden_layers,
            optimizer_function = opt.momentum_gradient_descent,
            optimizer_object = opt,
            weight_init = config.weight_init
        )

    if config.optimizer == "nesterov":

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
            gamma = 0.9
            ,training_set_size = 2048 #TEST
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
            epsilon = 1e-8,
            beta = 0.95
            ,training_set_size = 2048 #TEST
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
            beta = 0.9,
            beta2 = 0.999,
            epsilon = 1e-8
            ,training_set_size = 2048 #TEST
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
            beta = 0.9,
            beta2 = 0.999,
            epsilon = 1e-8
            ,training_set_size = 2048 #TEST
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

    # import time #TEST
    # time.sleep(120) #TEST

wb.agent(sweep_id, sweep, "cs22m056", "cs6910-assignment-1", 20)
