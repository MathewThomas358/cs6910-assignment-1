"""

Meta

"""

from functools import partial  # TEST
from typing import Callable, Any

import sys
import os

import numpy as np
import wandb as wb

from auxillary import Functions

RANDOM = "random"
XAVIER = "xavier"
DEBUG = True

if not DEBUG:
    sys.stdout = open(os.devnull, 'w', encoding="utf-8") #Disable print
else:
    sys.stdout = sys.__stdout__ # Enable print

# print = partial(print, flush=True)  # TEST

class Layer:
    """Layer"""

    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.activation_h: float | None = None
        self.pre_activation_a: float | None = None
        self.weights = None
        self.biases = None

        self.grad_loss_h = None
        self.grad_loss_a = None
        self.grad_loss_w = None
        self.grad_loss_b = None

        # For Nesterov, RMS, Momentum, Adam and Nadam
        self.v_weights = None
        self.v_biases = None

        # For Adam and Nadam
        self.m_weights = None
        self.m_biases = None

    def update(
        self,
        eta: float,
        gamma: float = 0,
        is_rms: bool = False,
        beta: float = None,
        eps: float = None,
        is_adam: bool = False,
        beta2: float = None,
        epoch: int = None,
        is_nadam: bool = False
    ) -> None:
        """Update"""

        if is_nadam:

            grad_loss_w_temp = self.grad_loss_w / (1 - np.power(beta, epoch))
            grad_loss_b_temp = self.grad_loss_b / (1 - np.power(beta, epoch))

            self.m_weights = (
                beta * self.m_weights +
                (1 - beta) * self.grad_loss_w
            )
            self.m_biases = (
                beta * self.m_biases +
                (1 - beta) * self.grad_loss_b
            )
            self.v_weights = (
                beta2 * self.v_weights +
                (1 - beta2) * self.grad_loss_w * self.grad_loss_w
            )
            self.v_biases = (
                beta2 * self.v_biases +
                (1 - beta2) * self.grad_loss_b * self.grad_loss_b
            )

            # TODO: ITP, make power computation once instead of twice
            m_w_hat = self.m_weights / (1 - np.power(beta, epoch + 1))
            m_b_hat = self.m_biases / (1 - np.power(beta, epoch + 1))

            v_w_hat = self.v_weights / (1 - np.power(beta2, epoch))
            v_b_hat = self.v_biases / (1 - np.power(beta2, epoch))

            m_w_temp = (1 - beta) * grad_loss_w_temp + beta * m_w_hat
            m_b_temp = (1 - beta) * grad_loss_b_temp + beta * m_b_hat

            self.weights -= (eta / np.sqrt(v_w_hat) + eps) * m_w_temp
            self.biases -= (eta / np.sqrt(v_b_hat) + eps) * m_b_temp
            return

        if is_adam:
            self.m_weights = (
                beta * self.m_weights +
                (1 - beta) * self.grad_loss_w
            )
            self.m_biases = (
                beta * self.m_biases +
                (1 - beta) * self.grad_loss_b
            )
            self.v_weights = (
                beta2 * self.v_weights +
                (1 - beta2) * self.grad_loss_w * self.grad_loss_w
            )
            self.v_biases = (
                beta2 * self.v_biases +
                (1 - beta2) * self.grad_loss_b * self.grad_loss_b
            )

            # TODO: ITP, make power computation once instead of twice
            m_w_hat = self.m_weights / (1 - np.power(beta, epoch + 1))
            m_b_hat = self.m_biases / (1 - np.power(beta, epoch + 1))

            v_w_hat = self.v_weights / (1 - np.power(beta2, epoch + 1))
            v_b_hat = self.v_biases / (1 - np.power(beta2, epoch + 1))

            self.weights -= (eta / np.sqrt(v_w_hat + eps)) * m_w_hat
            self.biases -= (eta / np.sqrt(v_b_hat + eps)) * m_b_hat
            return
        
        if is_rms:

            self.v_weights = (
                beta * self.v_weights +
                (1 - beta) * self.grad_loss_w * self.grad_loss_w
            )
            self.v_biases = (
                beta * self.v_biases +
                (1 - beta) * self.grad_loss_b * self.grad_loss_b
            )
            self.weights -= (eta / np.sqrt(self.v_weights + eps)
                             ) * self.grad_loss_w
            self.biases -= (eta / np.sqrt(self.v_biases + eps)
                            ) * self.grad_loss_b
            return

        weight_update = eta * self.grad_loss_w + gamma * self.v_weights
        bias_update = eta * self.grad_loss_b + gamma * self.v_biases
        self.weights = self.weights - weight_update
        self.biases = self.biases - bias_update
        self.v_biases = bias_update
        self.v_weights = weight_update

    def reset_grads(self):
        """Reset values"""
        self.grad_loss_h = np.zeros((self.grad_loss_h.shape[0], 1))
        self.grad_loss_a = np.zeros((self.grad_loss_a.shape[0], 1))
        self.grad_loss_w = np.zeros(
            (self.grad_loss_w.shape[0], self.grad_loss_w.shape[1]))
        self.grad_loss_b = np.zeros((self.grad_loss_b.shape[0], 1))

class Optimizers:
    """O"""

    def __init__(
        self,
        activation_function: Callable = None,
        loss_function: Callable = None,
        output_function: Callable = None,
        epochs: int = None,
        learning_rate: float = None,
        x_train: np.ndarray = None,
        y_train: np.ndarray = None,
        batch_size: int = None,
        gamma: float = None,
        epsilon: float = None,
        beta: float = None,
        beta2: float = None,
        training_set_size: int = None,
        x_val: np.ndarray = None,
        y_val: np.ndarray = None,
        l2_regpara: float = 0,
        is_sweeping: bool = False
    ):
        assert epochs is not None, "" #! TODO: Fill assertion messages
        assert activation_function is not None, "" #! TODO: Fill assertion messages
        assert loss_function is not None, "" #! TODO: Fill assertion messages
        assert output_function is not None, "" #! TODO: Fill assertion messages
        assert learning_rate is not None, "" #! TODO: Fill assertion messages
        assert x_train is not None, "" #! TODO: Fill assertion messages
        assert y_train is not None, "" #! TODO: Fill assertion messages
        assert x_val is not None, "" #! TODO: Fill assertion messages
        assert y_val is not None, "" #! TODO: Fill assertion messages

        self.activation_function = activation_function
        self.loss_function = loss_function
        self.output_function = output_function
        self.epochs = epochs
        self.eta = learning_rate
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.lamda = l2_regpara
        self.sweep_status = is_sweeping

        self.gamma = gamma
        self.beta = beta
        self.eps = epsilon
        self.beta_two = beta2

        if batch_size is None:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        if training_set_size is None:
            self.training_set_size = self.x_train.shape[0]
        else:
            self.training_set_size = training_set_size

    def gradient_descent(
        self,
        network: list[Layer],
        forward_propagation: Callable,
        back_propagation: Callable,
        is_stochastic: bool = True
    ):
        """SGD and Vanilla GD"""

        training_loss_cr = 0  # ! Change
        training_hits = 0  # ! Add
        norm = 0 # ! Add

        for j in range(self.epochs):

            training_loss_cr = 0  # ! Change
            training_hits = 0  # ! Add
            points_covered = 0

            arr = np.arange(self.x_train.shape[0])
            np.random.shuffle(arr)

            for i in range(self.training_set_size):

                points_covered += 1
                x_train_point = self.x_train[arr[i], :]
                y_train_label = self.y_train[arr[i], :]

                network[0].activation_h = np.expand_dims(x_train_point, axis=1)

                _, y_pred = forward_propagation(
                    network,
                    self.activation_function,
                    self.output_function
                )

                norm = self.__calculate_norm(network) # ! Add
                training_loss_cr += self.loss_function(
                    y_pred, y_train_label,
                    norm, self.lamda
                ) #! TODO: Add norm, lambda in other optimizers

                if np.argmax(y_pred.flatten()) == np.argmax(y_train_label):
                    training_hits += 1

                back_propagation(
                    network, y_pred, y_train_label,
                    Functions.get_grad(self.loss_function),
                    Functions.get_grad(self.activation_function),
                    self.lamda
                )

                if is_stochastic and points_covered == self.batch_size:
                    for k in range(1, network.shape[0]):
                        network[k].update(self.eta)
                    self.__reset_grads_in_network(network)
                    points_covered = 0

            if not is_stochastic:
                for i in range(1, network.shape[0]):
                    network[i].update(self.eta)
                self.__reset_grads_in_network(network)

            print("Epoch", j, "Training Loss:",
                  training_loss_cr / self.training_set_size) # ! Change

        training_accuracy = training_hits / self.training_set_size
        training_loss_cr = training_loss_cr / self.training_set_size

        print("Training accuracy:", training_accuracy)

        if self.sweep_status:
            evaluate_metrics_and_log(  # ! Add
            training_loss_cr,
            training_accuracy,
            self.x_val,
            self.y_val,
            network,
            forward_propagation,
            self.activation_function,
            self.output_function,
            self.loss_function,
            self.lamda,
            norm
        )

    def momentum_gradient_descent(
        self,
        network: list[Layer],
        forward_propagation: Callable,
        back_propagation: Callable,
        is_stochastic: bool = True
    ):
        """MGD"""

        assert self.gamma is not None, "gamma not provided"

        training_loss_cr = 0
        training_hits = 0
        norm = 0

        for j in range(self.epochs):

            training_loss_cr = 0
            training_hits = 0
            points_covered = 0

            arr = np.arange(self.x_train.shape[0])
            np.random.shuffle(arr)

            for i in range(self.training_set_size):

                points_covered += 1

                x_train_point = self.x_train[arr[i], :]
                y_train_label = self.y_train[arr[i], :]

                network[0].activation_h = np.expand_dims(x_train_point, axis=1)

                _, y_pred = forward_propagation(
                    network, self.activation_function, self.output_function)

                norm = self.__calculate_norm(network)
                training_loss_cr += self.loss_function(
                    y_pred, y_train_label,
                    norm, self.lamda
                )

                if np.argmax(y_pred.flatten()) == np.argmax(y_train_label):
                    training_hits += 1

                back_propagation(
                    network, y_pred, y_train_label,
                    Functions.get_grad(self.loss_function),
                    Functions.get_grad(self.activation_function),
                    self.lamda
                )

                if is_stochastic and points_covered == self.batch_size:
                    for k in range(1, network.shape[0]):
                        network[k].update(self.eta, self.gamma)
                    self.__reset_grads_in_network(network)
                    points_covered = 0

            if not is_stochastic:
                for i in range(1, network.shape[0]):
                    network[i].update(self.eta, self.gamma)
                self.__reset_grads_in_network(network)

            print("Epoch", j, "Training Loss:",
                  training_loss_cr / self.training_set_size)

        training_accuracy = training_hits / self.training_set_size
        training_loss_cr = training_loss_cr / self.training_set_size

        print("Training accuracy:", training_accuracy)

        if self.sweep_status:
            evaluate_metrics_and_log(
            training_loss_cr,
            training_accuracy,
            self.x_val, self.y_val,
            network, forward_propagation,
            self.activation_function,
            self.output_function,
            self.loss_function,
            self.lamda, norm
        )

    def nesterov_gradient_descent(
            self,
            network: list[Layer],
            forward_propagation: Callable,
            back_propagation: Callable,
            is_stochastic: bool = True
    ):
        """NGD"""

        assert self.gamma is not None, "gamma not provided"

        training_loss_cr = 0  # ! Change
        training_hits = 0  # ! Add
        norm = 0 # ! Add

        for j in range(self.epochs):

            training_loss_cr = 0  # ! Change
            training_hits = 0  # ! Add
            points_covered = 0

            arr = np.arange(self.x_train.shape[0])
            np.random.shuffle(arr)

            for i in range(self.training_set_size):

                points_covered += 1

                x_train_point = self.x_train[arr[i], :]
                y_train_label = self.y_train[arr[i], :]

                network[0].activation_h = np.expand_dims(x_train_point, axis=1)

                _, y_pred = forward_propagation(
                    network,
                    self.activation_function,
                    self.output_function
                )

                norm = self.__calculate_norm(network) # ! Add
                training_loss_cr += self.loss_function(
                    y_pred, y_train_label,
                    norm, self.lamda
                ) #! TODO: Add norm, lambda in other optimizers

                if np.argmax(y_pred.flatten()) == np.argmax(y_train_label):
                    training_hits += 1

                if points_covered == self.batch_size:
                    # This is a temporary update to the weights of the network.
                    # This updated weights serve as input while computing grads.
                    for k in range(1, network.shape[0]):
                        network[k].weights -= self.gamma * network[k].v_weights
                        network[k].biases -= self.gamma * network[k].v_biases

                back_propagation(
                    network, y_pred, y_train_label,
                    Functions.get_grad(self.loss_function),
                    Functions.get_grad(self.activation_function),
                    self.lamda
                )

                if is_stochastic and points_covered == self.batch_size:
                    for k in range(1, network.shape[0]):
                        # Reseting the weights to previous state before updating.
                        # We are adding the previously subtracted value back.
                        network[k].weights += self.gamma * network[k].v_weights
                        network[k].biases += self.gamma * network[k].v_biases
                        network[k].update(self.eta, self.gamma)
                    self.__reset_grads_in_network(network)
                    points_covered = 0

            if not is_stochastic:
                for i in range(1, network.shape[0]):
                    network[i].update(self.eta, self.gamma)
                self.__reset_grads_in_network(network)

            print("Epoch", j, "Training Loss:",
                  training_loss_cr / self.training_set_size)

        training_accuracy = training_hits / self.training_set_size
        training_loss_cr = training_loss_cr / self.training_set_size
        print("Training accuracy:", training_accuracy)

        if self.sweep_status:
            evaluate_metrics_and_log(  # ! Add
            training_loss_cr,
            training_accuracy,
            self.x_val,
            self.y_val,
            network,
            forward_propagation,
            self.activation_function,
            self.output_function,
            self.loss_function,
            self.lamda,
            norm
        )

    def rmsprop(
            self,
            network: list[Layer],
            forward_propagation: Callable,
            back_propagation: Callable,
            is_stochastic: bool = True
    ):
        """RMSProp"""

        assert self.eps is not None, "Epsilon not provided"
        assert self.beta is not None, "Beta not provided"

        training_loss_cr = 0  # ! Change
        training_hits = 0  # ! Add
        norm = 0 # ! Add

        for j in range(self.epochs):

            training_loss_cr = 0  # ! Change
            training_hits = 0  # ! Add
            points_covered = 0

            arr = np.arange(self.x_train.shape[0])
            np.random.shuffle(arr)

            for i in range(self.training_set_size):

                points_covered += 1

                x_train_point = self.x_train[arr[i], :]
                y_train_label = self.y_train[arr[i], :]

                network[0].activation_h = np.expand_dims(x_train_point, axis=1)

                _, y_pred = forward_propagation(
                    network,
                    self.activation_function,
                    self.output_function
                )

                norm = self.__calculate_norm(network) # ! Add
                training_loss_cr += self.loss_function(
                    y_pred, y_train_label,
                    norm, self.lamda
                )

                if np.argmax(y_pred.flatten()) == np.argmax(y_train_label):
                    training_hits += 1

                back_propagation(
                    network, y_pred, y_train_label,
                    Functions.get_grad(self.loss_function),
                    Functions.get_grad(self.activation_function),
                    self.lamda
                )

                if is_stochastic and points_covered == self.batch_size:
                    for k in range(1, network.shape[0]):
                        network[k].update(
                            self.eta, is_rms=True, beta=self.beta, eps=self.eps)
                    self.__reset_grads_in_network(network)
                    points_covered = 0

            if not is_stochastic:
                for i in range(1, network.shape[0]):
                    network[i].update(self.eta, self.gamma)
                self.__reset_grads_in_network(network)

            print("Epoch", j, "Training Loss:",
                  training_loss_cr / self.training_set_size)

        training_accuracy = training_hits / self.training_set_size
        training_loss_cr = training_loss_cr / self.training_set_size
        print("Training accuracy:", training_accuracy)

        if self.sweep_status:
            evaluate_metrics_and_log(  # ! Add
            training_loss_cr,
            training_accuracy,
            self.x_val,
            self.y_val,
            network,
            forward_propagation,
            self.activation_function,
            self.output_function,
            self.loss_function,
            self.lamda,
            norm
        )

    def adam(
            self,
            network: list[Layer],
            forward_propagation: Callable,
            back_propagation: Callable,
            is_stochastic: bool = True
    ):
        """Adam"""
        assert self.eps is not None, "Epsilon not provided"
        assert self.beta is not None, "Beta not provided"
        assert self.beta_two is not None, "Beta 2 not provided"

        training_loss_cr = 0  # ! Change
        training_hits = 0  # ! Add
        norm = 0 # ! Add

        for j in range(self.epochs):

            training_loss_cr = 0  # ! Change
            training_hits = 0  # ! Add
            points_covered = 0
            update_count = 0

            arr = np.arange(self.x_train.shape[0])
            np.random.shuffle(arr)

            for i in range(self.training_set_size):

                points_covered += 1
                x_train_point = self.x_train[arr[i], :]
                y_train_label = self.y_train[arr[i], :]

                network[0].activation_h = np.expand_dims(x_train_point, axis=1)

                _, y_pred = forward_propagation(
                    network,
                    self.activation_function,
                    self.output_function
                )

                norm = self.__calculate_norm(network) # ! Add
                training_loss_cr += self.loss_function(
                    y_pred, y_train_label,
                    norm, self.lamda
                )

                if np.argmax(y_pred.flatten()) == np.argmax(y_train_label):
                    training_hits += 1

                back_propagation(
                    network, y_pred, y_train_label,
                    Functions.get_grad(self.loss_function),
                    Functions.get_grad(self.activation_function),
                    self.lamda
                )

                if is_stochastic and points_covered == self.batch_size:
                    update_count += 1
                    for k in range(1, network.shape[0]):
                        network[k].update(
                            self.eta,
                            is_adam = True,
                            beta = self.beta,
                            beta2 = self.beta_two,
                            eps = self.eps,
                            epoch = update_count
                        )
                    self.__reset_grads_in_network(network)
                    points_covered = 0

            if not is_stochastic:
                for i in range(1, network.shape[0]):
                    network[i].update(self.eta)
                self.__reset_grads_in_network(network)

            print("Epoch", j, "Training Loss:",
                  training_loss_cr / self.training_set_size)

        training_accuracy = training_hits / self.training_set_size
        training_loss_cr = training_loss_cr / self.training_set_size
        print("Training accuracy:", training_accuracy)

        if self.sweep_status:
            evaluate_metrics_and_log(
            training_loss_cr,
            training_accuracy,
            self.x_val, self.y_val,
            network,
            forward_propagation,
            self.activation_function,
            self.output_function,
            self.loss_function,
            self.lamda,
            norm
        )

    def nadam(
            self,
            network: list[Layer],
            forward_propagation: Callable,
            back_propagation: Callable,
            is_stochastic: bool = True
    ):
        """Nadam"""

        # Reference: https://cs229.stanford.edu/proj2015/054_report.pdf
        #
        # As per the report, we are supposed to use a "warming schedule",
        # for momentum and we are supposed to divide our gradients by this
        # factor (Refer: Algorithm 8, Pg 3). Instead of a "warming schedule"
        # we will just raise momentum to the power of number of updates. 

        assert self.eps is not None, "Epsilon not provided"
        assert self.beta is not None, "Beta not provided"
        assert self.beta_two is not None, "Beta 2 not provided"

        training_loss_cr = 0  # ! Change
        training_hits = 0  # ! Add
        norm = 0 # ! Add

        for j in range(self.epochs):

            training_loss_cr = 0  # ! Change
            training_hits = 0  # ! Add
            points_covered = 0
            update_count = 0

            arr = np.arange(self.x_train.shape[0])
            np.random.shuffle(arr)

            for i in range(self.training_set_size):

                points_covered += 1
                x_train_point = self.x_train[arr[i], :]
                y_train_label = self.y_train[arr[i], :]

                network[0].activation_h = np.expand_dims(x_train_point, axis=1)

                _, y_pred = forward_propagation(
                    network,
                    self.activation_function,
                    self.output_function
                )

                norm = self.__calculate_norm(network) # ! Add
                training_loss_cr += self.loss_function(
                    y_pred, y_train_label,
                    norm, self.lamda
                )

                if np.argmax(y_pred.flatten()) == np.argmax(y_train_label):
                    training_hits += 1

                back_propagation(
                    network, y_pred, y_train_label,
                    Functions.get_grad(self.loss_function),
                    Functions.get_grad(self.activation_function),
                    self.lamda
                )

                if is_stochastic and points_covered == self.batch_size:
                    update_count += 1
                    for k in range(1, network.shape[0]):
                        network[k].update(
                            self.eta,
                            is_nadam = True,
                            beta = self.beta,
                            beta2 = self.beta_two,
                            eps = self.eps,
                            epoch = update_count
                        )
                    self.__reset_grads_in_network(network)
                    points_covered = 0

            if not is_stochastic:
                for i in range(1, network.shape[0]):
                    network[i].update(self.eta)
                self.__reset_grads_in_network(network)

            print("Epoch", j, "Training Loss:",
                  training_loss_cr / self.training_set_size)

        training_accuracy = training_hits / self.training_set_size
        training_loss_cr = training_loss_cr / self.training_set_size
        print("Training accuracy:", training_accuracy)

        if self.sweep_status:
            evaluate_metrics_and_log(
            training_loss_cr,
            training_accuracy,
            self.x_val, self.y_val,
            network, forward_propagation,
            self.activation_function,
            self.output_function,
            self.loss_function,
            self.lamda, norm
        )



    def __reset_grads_in_network(self, network: np.ndarray[Layer, Any]):

        for i in range(1, network.shape[0]):
            network[i].reset_grads()

    def __calculate_norm(self, network: list[Layer]) -> float:
        """Calculates the norm of all the weights in the network"""

        norm = 0
        for i in range(1, network.shape[0]):
            norm += np.sum(np.square(network[i].weights))
        return norm

class NeuralNetwork:
    """Neural Network implementation"""

    def __init__(self,
                 input_size: int = 28 * 28,
                 hidden_size: list[int] = None,
                 output_size: int = 10,
                 is_hidden_layer_size_variable=False,
                 no_of_hidden_layers: int = 4,
                 optimizer_function: Callable = None,
                 optimizer_object: Optimizers = None,
                 weight_init: str = XAVIER
                 ):
        """

        hidden_size: If the number of neurons in each in layer is the same,
                    then the list will only contain one elements. If the number varies
                    from layer to layer, then provide the entire list.
        """

        assert hidden_size is not None, "Hidden size list not provided"
        assert optimizer_function is not None, "Optimizer not provided"
        assert optimizer_object is not None, "Optimzer object not provided"
        self.optimizer: Optimizers = optimizer_function
        self.optimizer_object = optimizer_object
        self.activation_function = optimizer_object.activation_function
        self.loss_function = optimizer_object.loss_function
        self.output_function = optimizer_object.output_function
        self.weight_init = weight_init

        if is_hidden_layer_size_variable:
            self.sizes = [input_size] + hidden_size + [output_size] 
            self.total_layers = len(self.sizes)
        else:
            self.sizes = [input_size]
            for _ in range(no_of_hidden_layers):
                self.sizes = self.sizes + [hidden_size[0]]
            self.sizes = self.sizes + [output_size]
            self.total_layers = no_of_hidden_layers + 2

        assert len(self.sizes) == self.total_layers

        self.network = np.empty((self.total_layers, ), dtype=Layer)

        self.network[0] = Layer(0)
        self.network[0].pre_activation_a = np.zeros(
            (self.sizes[0], 1), dtype=np.float64)

        for i in range(1, self.total_layers):

            layer = Layer(i)
            self.__weight_init(layer, i)
            layer.grad_loss_a = np.zeros((self.sizes[i], 1), dtype=np.float64)
            layer.grad_loss_h = np.zeros((self.sizes[i], 1), dtype=np.float64)
            layer.grad_loss_b = np.zeros((self.sizes[i], 1), dtype=np.float64)
            layer.grad_loss_w = np.zeros(
                (self.sizes[i], self.sizes[i-1]), dtype=np.float64)

            layer.v_biases = np.zeros((self.sizes[i], 1), dtype=np.float64)
            layer.v_weights = np.zeros(
                (self.sizes[i], self.sizes[i-1]), dtype=np.float64)
            
            layer.m_biases = np.zeros((self.sizes[i], 1), dtype=np.float64)
            layer.m_weights = np.zeros(
                (self.sizes[i], self.sizes[i-1]), dtype=np.float64
            )

            self.network[i] = layer

    def __weight_init(self, layer: Layer, i: int):

        if self.weight_init == RANDOM:

            if self.optimizer_object.activation_function is Functions.ActivationFunctions.relu:
                factor = np.sqrt(2 / (self.sizes[i] + self.sizes[i-1]))
                layer.weights = factor * \
                    np.random.randn(self.sizes[i], self.sizes[i-1])
                layer.biases = np.zeros((self.sizes[i], 1))
            else:
                layer.biases = np.random.rand(int(self.sizes[i]), 1)
                layer.weights = np.random.rand(
                    int(self.sizes[i]), int(self.sizes[i-1]))

        if self.weight_init == XAVIER:

            if self.optimizer_object.activation_function is Functions.ActivationFunctions.relu:
                factor = np.sqrt(2 / self.sizes[i-1])
                layer.weights = factor * \
                    np.random.rand(int(self.sizes[i]), int(self.sizes[i-1]))
                layer.biases = factor * np.random.rand(int(self.sizes[i]), 1)
            else:
                factor = 2 / np.sqrt(self.sizes[i-1])
                layer.weights = factor * \
                    (-0.5 +
                     np.random.rand(int(self.sizes[i]), int(self.sizes[i-1])))
                layer.biases = factor * np.random.rand(int(self.sizes[i]), 1)

    def feed_forward_propagation(self,
                                 layers: list[Layer],
                                 activation_function: Callable,
                                 output_activation_function: Callable,
                                 ):
        """FFWD"""

        for i in range(1, self.total_layers - 1):

            layers[i].pre_activation_a = (
                layers[i].biases +
                np.dot(layers[i].weights, layers[i-1].activation_h)
            )

            layers[i].activation_h = activation_function(
                layers[i].pre_activation_a
            )

        layers[self.total_layers - 1].pre_activation_a = (
            layers[self.total_layers - 1].biases +
            np.dot(
                layers[self.total_layers - 1].weights,
                layers[self.total_layers - 2].activation_h
            )
        )
        y_pred = output_activation_function(
            layers[self.total_layers - 1].pre_activation_a)

        return layers, y_pred

    def back_propagation(
        self,
        layers: list[Layer],
        y_pred: np.ndarray,
        y_true: np.ndarray,
        gradient_loss_function: Callable,
        gradient_activation_function: Callable,
        lamda: float
    ):
        """BP"""
        y_true = np.expand_dims(y_true, axis=1)
        layers[self.total_layers -
               1].grad_loss_a = gradient_loss_function(y_pred, y_true)

        for i in range(self.total_layers - 1, 0, -1):

            layers[i].grad_loss_w += np.dot(
                layers[i].grad_loss_a,
                layers[i-1].activation_h.T
            ) + lamda * layers[i].weights

            assert layers[i].grad_loss_w is not None, i

            layers[i].grad_loss_b += layers[i].grad_loss_a

            layers[i-1].grad_loss_h = np.dot(layers[i].weights.T,
                                             layers[i].grad_loss_a)
            layers[i-1].grad_loss_a = (
                layers[i-1].grad_loss_h *
                gradient_activation_function(
                    layers[i-1].pre_activation_a
                )
            )

        return layers

    def print(self):
        """Print"""

        for i in range(1, self.network.shape[0]):
            print("W", i, self.network[i].weights.flatten())
            print("B", i, self.network[i].biases.flatten())

    def train(self):
        """Train"""
        return self.optimizer(self.network, self.feed_forward_propagation, self.back_propagation)

    def predict(self, x_test):
        """Predict the class of the given x"""
        self.network[0].activation_h = np.expand_dims(x_test, axis=1)
        _, y_pred = self.feed_forward_propagation(
            self.network, self.activation_function, self.output_function)
        
        assert 1+1e-9 > np.sum(y_pred) > 1-1e-9 #? Stupid assertion? Made sense when going crazy
        return y_pred.flatten()

def evaluate_metrics_and_log(
    training_loss: float,
    training_accuracy: float,
    x_val: np.ndarray,
    y_val: np.ndarray,
    network: np.ndarray,
    forward_propagation: Callable,
    activation_function: Callable,
    output_function: Callable,
    loss_function: Callable,
    lamda: float,
    norm: float
):
    """Used to evaluate the training and validation accuracies and losses"""
    #! TODO: Move to auxillary

    validation_hits = 0
    validation_loss = 0

    for i in range(x_val.shape[0]):

        x_val_point = x_val[i, :]
        y_val_label = y_val[i, :]

        network[0].activation_h = np.expand_dims(x_val_point, axis=1)

        _, y_pred = forward_propagation(
            network, activation_function, output_function)
        
        if np.argmax(y_pred.flatten()) == np.argmax(y_val_label):
            validation_hits += 1

        validation_loss += loss_function(y_pred, y_val_label, norm, lamda) #! TODO

    validation_accuracy = validation_hits / x_val.shape[0]
    validation_loss = validation_loss / x_val.shape[0]

    metrics = {
        "training_accuracy" : float(training_accuracy),
        "training_loss" : float(training_loss),
        "validation_accuracy" : float(validation_accuracy),
        "validation_loss" : float(validation_loss)
    }

    wb.log(metrics)
    print(metrics)


def main():
    """Main"""

    from data import get_data
    train, _, val = get_data()

    optimizer = Optimizers(
        Functions.ActivationFunctions.tanh,
        Functions.LossFunctions.cross_entropy,
        Functions.softmax,
        10, 1e-5, train[0], train[1], 128,
        x_val = val[0], y_val = val[1],
        training_set_size=10240,
        is_sweeping=False,
        l2_regpara=0.05,
        epsilon = 1e-8,
        beta = 0.9,
        beta2 = 0.999
    )

    nn = NeuralNetwork(  # pylint: disable=C0103
        28 * 28, [256, 128, 64], 10,
        True, 5,
        optimizer.nadam,
        optimizer
    )

    nn.train()

    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # x_val = x_train[:6000]
    # y_val = y_train[:6000]
    # x_train = x_train[:54000]
    # y_train = y_train[:54000]

    # x_train = x_train.reshape(x_train.shape[0], 28*28)
    # x_val = x_val.reshape(x_val.shape[0], 28*28)
    # x_test = x_test.reshape(x_test.shape[0], 28*28)

    # x_train = x_train / 255.0
    # x_test = x_test / 255.0
    # x_val = x_val / 255.0

    # y_train = create_one_hot_vector(y_train)
    # y_test = create_one_hot_vector(y_test)
    # y_val = create_one_hot_vector(y_val)

    # optimizer = Optimizers(
    #     Functions.ActivationFunctions.tanh,
    #     Functions.LossFunctions.cross_entropy,
    #     Functions.softmax,
    #     15, 1e-5, x_train, y_train, 128,
    #     beta=0.9, epsilon=1e-8, beta2 = 0.999  # , training_set_size=10800
    # )

    # nn = NeuralNetwork(  # pylint: disable=C0103
    #     28 * 28, [128], 10,
    #     False, 4,
    #     optimizer.adam,
    #     optimizer
    # )

    # nn.train()

    count = 0
    temp = 10

    for i in range(val[0].shape[0]):

        y_pred = nn.predict((val[0])[i, :])
        final = np.zeros_like(y_pred)
        final[np.argmax(y_pred)] = 1
        if temp > 0:
            # tq2=" ".join([f"{q:1.3f}" for q in x_val[i,:]])
            t_q = " ".join([f"{q:1.3f}" for q in y_pred])
            print(f"{temp}: {t_q} - {(val[1])[i,:]}")
            temp -= 1

        # print("Pred", y_pred)
        if ((val[1])[i, :] == final).all():
            count += 1

    print(100 * count/(val[0]).shape[0])

    # TODO
    #// ! Add evaluate metrics for all optimizers
    #! TODO: Recheck all algorithms for correctness
    #! Search for TODOs across files and find all the red ones
    #! Make sure that both stochastic and stochastic updates are the same
    #! README.md
    #! PyDocs
    # // TODO: Batch sizes things -> 54000 images to be processed but in batches -> done
    #! TODO: All optimizers should have the same signature, ig
    # // TODO: If gradient descent, then don't take gamma -> Done using assertion that gamma is None
    # losses = []
    # for k,l in zip([32,64,128],[5,4,3]):

    #     for j in [1e-3, 5e-3, 1e-4, 5e-4, 1e-5]:

    #         for (x, y) in zip([10, 15, 20, 25, 20],[4096, 3072, 2048, 1500, 4096]):

    #             import datetime
    #             print("START", datetime.datetime.now())

    #             optimizer = Optimizers(
    #                 Functions.ActivationFunctions.tanh,
    #                 Functions.LossFunctions.cross_entropy,
    #                 Functions.softmax,
    #                 int(x), j, x_train, y_train, int(y),
    #                 gamma = k / 10
    #             )

    #             nn = NeuralNetwork(                                     #pylint: disable=C0103
    #                 28 * 28, [k], 10,
    #                 False, l,
    #                 optimizer.gradient_descent,
    #                 optimizer
    #             )

    #             nn.train()

    #             count = 0
    #             temp = 5

    #             for i in range(x_val.shape[0]):

    #                 y_pred = nn.predict(x_val[i,:])
    #                 final = np.zeros_like(y_pred)
    #                 final[np.argmax(y_pred)] = 1
    #                 if temp > 0:
    #                     # tq2=" ".join([f"{q:1.3f}" for q in x_val[i,:]])
    #                     t_q= " ".join([f"{q:1.3f}" for q in y_pred])
    #                     print(f"{temp}: {t_q} - {y_val[i,:]}")
    #                     temp -= 1

    #                 # print("Pred", y_pred)
    #                 if (y_val[i,:] == final).all():
    #                     count += 1

    #             print(k, l, j, x, y, 100 * count/x_val.shape[0], flush=True)
    #             losses.append((k, l, j, x, y, 100 * count/x_val.shape[0]))
    #             print("END", datetime.datetime.now())

    # for loss in losses:
    #     print(loss)


if __name__ == "__main__":
    main()
