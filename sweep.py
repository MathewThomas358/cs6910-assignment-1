"""
    Sweep
"""

import wandb as wb

sweep_conf = {

    'method' : 'bayes',
    'metric' : {
      'name' : 'accuracy',
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
            'values': [ 'sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
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
    pass