"""
CS6910 - Assignment 1

Based on the parameters given as arguments or based on the default values,
initiates a neural network training sequence and then makes a prediction
on the test set and report the accuracy on the test set.

@author: cs22m056
"""

import argparse as ap

from sweep import init

def main():
    """
    The most important function. The Alpha. The Main function.
    """

    parser = ap.ArgumentParser(
        description= "Utility for training and predicting MNIST-like datesets"
    )
    parser.add_argument("-wp", "--wandb_project", type=str, default="cs6910-assignment-1")
    parser.add_argument("-we", "--wandb_entity", type=str, default="cs22m056")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default=15)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy")
    parser.add_argument("-o", "--optimizer", type=str, default="adam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-beta", "--beta", type=float, default=0.95)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0)
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4)
    parser.add_argument("-sz", "--hidden_size", type=int, default=32)
    parser.add_argument("-a", "--activation", type=str, default="tanh")
    args = parser.parse_args()

    sweep_conf = {

            'method' : 'bayes',
            'metric' : {
            'name' : 'validation_accuracy',
            'goal' : 'maximize'   
            },
            'parameters': {
                'epochs': {
                    'values': [args.epochs]
                },
                'is_hidden_layer_size_variable': {
                    'values': [False]
                },
                'no_of_hidden_layers': {
                    'values': [args.num_layers]
                },
                'hidden_size_1': {
                    'values': [args.hidden_size]
                },
                'l2_regpara': {
                    'values': [args.weight_decay]
                },
                'learning_rate': {
                    'values': [args.learning_rate] 
                },
                'optimizer': {
                    'values': [args.optimizer]
                },
                'batch_size' : {
                    'values':[args.batch_size]
                },
                'weight_init': {
                    'values': [args.weight_init]
                },
                'activation': {
                    'values': [args.activation]
                }
            }
        }

    init(
        wandb_project = args.wandb_project,
        wandb_entity = args.wandb_entity,
        sweep_conf = sweep_conf,
        sweep_count = 1,
        additional_params = args,
        is_invoked_from_train = True
    )

if __name__ == "__main__":
    main()
