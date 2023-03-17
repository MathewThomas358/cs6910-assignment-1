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
        'no_of_hidden_layers': {
            'values': [3, 4, 5]
        },
        'hidden_size': {
            'values': [32, 64, 128, 256]
        },
        'l2_regpara': {
            'values': [0, 0.5,  0.05, 0.005]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4, 1e-5] 
        },
        'optimizer': {
            'values': [ 'sgd' ] #, 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
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

def sweep():
    """#! TODO"""

    conf = {}

    wb.init(config = conf, resume = "auto")
    config = wb.config
    wb.run.name = (
        "op_" + str(config.optimizer) + 
        "_nh_" + str(config.no_of_hidden_layers) + 
        "_hl_" + str(config.hidden_size) +
        "_bs_" + str(config.batch_size) + 
        "_ac_" + str(config.batch_size)
    )

    train, val, _ = get_data()
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
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size = config.hidden_size,
            no_of_hidden_layers = config.no_of_hidden_layers,
            optimizer_function = opt.gradient_descent,
            optimizer_object = opt,
            weight_init = config.weight_init
        )

        nn.train()


wb.agent(sweep_id, sweep, "cs22m056", "cs6910-assignment-1", 1)
