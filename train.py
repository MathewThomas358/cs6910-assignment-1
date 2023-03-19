"""
#! TODO
"""

import argparse as ap

from network import Optimizers, NeuralNetwork
from data import get_data
from auxillary import map_functions, Functions, map_optimizer

def main():
    """The main function"""

    parser = ap.ArgumentParser(
        description= "Utility for training and predicting MNIST-like datesets"
    )
    parser.add_argument("-wp", "--wandb_project", type=str, default="cs6910-assignment-1")
    parser.add_argument("-we", "--wandb_entity", type=str, default="cs22m056")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default="mnist") # TODO Default
    parser.add_argument("-b", "--batch_size", type=int, default="mnist") # TODO Default
    parser.add_argument("-l", "--loss", type=str, default="mnist") # TODO Default
    parser.add_argument("-o", "--optimizer", type=str, default="mnist") # TODO Default
    parser.add_argument("-lr", "--learning_rate", type=float, default="mnist") # TODO Default
    parser.add_argument("-m", "--momentum", type=float, default="mnist") # TODO Default
    parser.add_argument("-beta", "--beta", type=float, default="mnist") # TODO Default
    parser.add_argument("-beta1", "--beta1", type=float, default="mnist") # TODO Default
    parser.add_argument("-beta2", "--beta2", type=float, default="mnist") # TODO Default
    parser.add_argument("-eps", "--epsilon", type=float, default="mnist") # TODO Default
    parser.add_argument("-w_d", "--weight_decay", type=float, default="mnist") # TODO Default
    parser.add_argument("-w_i", "--weight_init", type=str, default="mnist") # TODO Default
    parser.add_argument("-nhl", "--num_layers", type=int, default="mnist") # TODO Default
    parser.add_argument("-sz", "--hidden_size", type=int, default="mnist") # TODO Default
    parser.add_argument("-a", "--activation", type=str, default="mnist") # TODO Default
    args = parser.parse_args()

    train, test, val = get_data(args.dataset)

    if args.optimizer == "sgd":

        opt = Optimizers(
            map_functions(args.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            args.epochs,
            args.learning_rate,
            train[0], train[1],
            args.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = args.weight_decay,
            is_sweeping = True
            #,training_set_size = 8196 #TEST
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size =args.hidden_size,
            is_hidden_layer_size_variable = False,
            no_of_hidden_layers = args.num_layers,
            optimizer_function = map_optimizer(args.optimizer, opt),
            optimizer_object = opt,
            weight_init = args.weight_init
        )

    if args.optimizer == "momentum":

        opt = Optimizers(
            map_functions(args.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            args.epochs,
            args.learning_rate,
            train[0], train[1],
            args.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = args.weight_decay,
            is_sweeping = True,
            gamma = 0.9
            #,training_set_size = 8196 #TEST
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size =args.hidden_size,
            is_hidden_layer_size_variable = False,
            no_of_hidden_layers = args.num_layers,
            optimizer_function = map_optimizer(args.optimizer, opt),
            optimizer_object = opt,
            weight_init = args.weight_init
        )

    if args.optimizer == "nesterov":

        opt = Optimizers(
            map_functions(args.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            args.epochs,
            args.learning_rate,
            train[0], train[1],
            args.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = args.weight_decay,
            is_sweeping = True,
            gamma = 0.9
            #,training_set_size = 8196 #TEST
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size =args.hidden_size,
            is_hidden_layer_size_variable = False,
            no_of_hidden_layers = args.num_layers,
            optimizer_function = map_optimizer(args.optimizer, opt),
            optimizer_object = opt,
            weight_init = args.weight_init
        )

    if args.optimizer == "rmsprop":

        opt = Optimizers(
            map_functions(args.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            args.epochs,
            args.learning_rate,
            train[0], train[1],
            args.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = args.weight_decay,
            is_sweeping = True,
            epsilon = 1e-8,
            beta = 0.95
            #,training_set_size = 8196 #TEST
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size =args.hidden_size,
            is_hidden_layer_size_variable = False,
            no_of_hidden_layers = args.num_layers,
            optimizer_function = map_optimizer(args.optimizer, opt),
            optimizer_object = opt,
            weight_init = args.weight_init
        )

    if args.optimizer == "adam":

        opt = Optimizers(
            map_functions(args.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            args.epochs,
            args.learning_rate,
            train[0], train[1],
            args.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = args.weight_decay,
            is_sweeping = True,
            beta = 0.9,
            beta2 = 0.999,
            epsilon = 1e-8
            #,training_set_size = 8196 #TEST
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size =args.hidden_size,
            is_hidden_layer_size_variable = False,
            no_of_hidden_layers = args.num_layers,
            optimizer_function = map_optimizer(args.optimizer, opt),
            optimizer_object = opt,
            weight_init = args.weight_init
        )

    if args.optimizer == "nadam":

        opt = Optimizers(
            map_functions(args.activation),
            Functions.LossFunctions.cross_entropy,
            Functions.softmax,
            args.epochs,
            args.learning_rate,
            train[0], train[1],
            args.batch_size,
            x_val = val[0],
            y_val = val[1],
            l2_regpara = args.weight_decay,
            is_sweeping = True,
            beta = 0.9,
            beta2 = 0.999,
            epsilon = 1e-8
            #,training_set_size = 8196 #TEST
        )

        nn = NeuralNetwork( #pylint: disable=C0103
            hidden_size =args.hidden_size,
            is_hidden_layer_size_variable = False,
            no_of_hidden_layers = args.num_layers,
            optimizer_function = map_optimizer(args.optimizer, opt),
            optimizer_object = opt,
            weight_init = args.weight_init
        )
    
    nn.train()

if __name__ == "__main__":
    main()