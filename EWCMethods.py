# fmt: off
from copy import deepcopy
from enum import Enum
from typing import List, Union
import numpy as np
import os

from SequentialTask import SequentialTask
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class EWC_Method(Enum):
    NONE = 1,
    WEIGHT_DECAY = 2
    SIGN_FLIPPING = 3
    FISHER_MATRIX = 4
class WeightTrackingCallback(tf.keras.callbacks.Callback):
    """
    A callback to track weights during training.
    Weights can be tracked by keeping a direct history, checking sign flipping, etc...
    Each tracking method is implemented in a child class
    """

    def __init__(self, model: tf.keras.models.Model):
        """
        Create a new instance of this call back.

        Parameters:
        model: tf.keras.models.Model
                        The model to track weights of.
                        Must be passed so lifecycle hooks can reference model!
        """

        super().__init__()
        self.model = model
        # Okay, strap in
        # It appears that Tensorflow, in the backend, does NOT handle memory 
        # addresses correctly when it comes to models built from other models
        #
        # In particular, it appears that whatever model is "in use" overwrites 
        # any references to the previous model.
        # For our application, that means base_model (with, say, 2 layers) has 
        # 2 layers at init time, but that same reference will now point to
        # new_model (with, say, 4 layers) at callback time! For this reason,
        # A work around is to store a direct list of references to the model
        # layers to be used instead of self.model.layers
        # 
        # It appears this issue persists with other properties, i.e. 
        # the entire reference changes!
        self.model_layers = model.layers

class SignFlippingTracker(WeightTrackingCallback):
    """
    Track the weights of a network by checking the sign of those weights after each epoch
    Count how often each weight flips sign
    """

    class MeasurePeriod(Enum):
        BATCH_END = 1,
        EPOCH_END = 2

    def __init__(self, model: tf.keras.models.Model,
                 measure_period: MeasurePeriod = MeasurePeriod.BATCH_END):
        """
        Create a new instance of this call back.

        Parameters:
            model: tf.keras.models.Model
                The model to track weights of.
                Must be passed so lifecycle hooks can reference model!

            measure_period: MeasurePeriod
                How often to measure the weights for updates
        """

        super().__init__(model)
        self.measure_period = measure_period
        self.sign_changes = []
        self.stored_weights = []
        self.reset_tracking()

    def reset_tracking(self):
        for layer in self.model_layers:
            layer_weights = []
            layer_sign_change = []
            for weight in layer.weights:
                layer_weights.append(weight)
                layer_sign_change.append(tf.zeros_like(weight))
            self.stored_weights.append(layer_weights)
            self.sign_changes.append(layer_sign_change)

    def measure_weight_changes(self):
        updated_weights = []
        for layer_index, layer in enumerate(self.model_layers):
            layer_weights = []
            curr_layer = self.stored_weights[layer_index]
            for weight_index, weight in enumerate(layer.weights):
                layer_weights.append(weight)
                sign_change = tf.sign(weight*curr_layer[weight_index]) == -1
                self.sign_changes[layer_index][weight_index] += tf.cast(sign_change, dtype=tf.float32)
            updated_weights.append(layer_weights)
        self.stored_weights = updated_weights

    def on_train_begin(self, logs=None):
        self.reset_tracking()

    def on_batch_end(self, epoch, logs=None):
        if self.measure_period == self.MeasurePeriod.BATCH_END:
            self.measure_weight_changes()

    def on_epoch_end(self, epoch, logs=None):
        if self.measure_period == self.MeasurePeriod.EPOCH_END:
            self.measure_weight_changes()

class FisherInformationMatrixCalculator(tf.keras.callbacks.Callback):
    """
    A callback to track tasks and calculate the fisher information matrix when called
    This class probably doesn't need to be a callback, but it also doesn't hurt if it's needed
    """

    def __init__(self, tasks: List[SequentialTask], samples: int=-1):
        """
        Create a new fisher information callback - creating a new fisher information matrix
        at the end of each task

        Note! This callback will track gradients for the ENTIRE network,
        not only the "shared" section. See the EWC_Term class for why this is not a problem

        Parameters:
            tasks: A list of sequential tasks 
            samples: The number of samples to take from the training data to construct the matrix
                Defaults to -1, meaning take the entire dataset
        """
        super().__init__()
        self.tasks: List[SequentialTask] = tasks
        self.current_task_index: int = 0
        self.samples: int = samples
        self.fisher_matrices: List[List[List[tf.Tensor]]] = []

    def on_train_end(self, logs=None):
        # Do fisher calculation
        print(f"{'-'*80}")
        print("STARTING FISHER CALCULATION")

        current_task = self.tasks[self.current_task_index]
        samples = self.samples if self.samples!=-1 else current_task.training_batches

        # Notice for this section we work with flattened arrays until the end (for speed)
        # init variance to be a zero matrix like all weight tensors of task model
        variance = [tf.zeros_like(tensor) for tensor in current_task.model.weights]
        # Noting gradients, apply model to samples
        step = 0
        # Each loop we get the gradients for a new sample
        # And update the variance to account for the new sample
        for x, _ in current_task.training_data.take(samples):
            outputs = []
            # Track the gradient over the log likelihood
            with tf.GradientTape() as tape:
                outputs = current_task.model(x)
                log_likelihood = tf.math.log(outputs)
            # Finally actually take the gradient and update the variance accordingly
            gradients = tape.gradient(log_likelihood, current_task.model.weights)
            variance = [var + (grad ** 2) for var, grad in zip(variance, gradients)]

            # Show the current step to keep user updated
            print(f"STEP: {step:05}/{samples:05}", end='\r')
            step+=1
        # Finally - Fisher matrix is found by variance divided by number of samples
        fisher_diagonal = [tensor / samples for tensor in variance]

        # Now we can stop working with flat arrays and convert to layer-collected format
        # But first, convert with these ugly loops
        current_fisher = []
        index = 0
        for layer_index, layer in enumerate(current_task.model.layers):
            current_layer = []
            for tensor_index, tensor in enumerate(layer.weights):
                current_layer.append(deepcopy(fisher_diagonal[index]))
                index += 1
            current_fisher.append(current_layer)
        self.fisher_matrices.append(current_fisher)

        print("FINISHED FISHER CALCULATION")
        print(f"{'-'*80}")

        # Move to next task
        self.current_task_index += 1

class EWC_Term():
    """
    Representation of a single EWC_Term
    Collects together all EWC term ideas (lambda, optimal weights, omega...)
    Also exposes loss function for use

    Note the loss function only loops over the OPTIMAL weights given
    So if a larger omega matrix is given (e.g. Fisher over all weights) this is okay!
    The extra omega matrix is ignored
    """


    def __init__(self,
                 lam: float,
                 optimal_weights: List[List[np.ndarray]],
                 omega_matrix: List[List[np.ndarray]]):
        """
        A single EWC term for model training

        Parameters:
            lam: float
                The importance of this EWC term. 

            optimal_weights: List[List[np.ndarray]]
                The optimal weights of the model after training.
                Can be found by model.weights
                Note! Should only be the *shared* weights 

            omega_matrix: List[List[np.ndarray]]
                The weight importance matrix for this term.
                Should have the same dimensions (in every way) as 
                optimal_weights
        """

        self.lam = lam
        self.optimal_weights = deepcopy(optimal_weights)
        self.omega_matrix = deepcopy(omega_matrix)

    def calculate_loss(self, model_layers: List[tf.keras.layers.Layer]):
        loss = 0
        for layer_index, layer in enumerate(model_layers):
            # Note in zip function - if omega is longer than optimal weights the excess omega is ignored
            # This may be an issue if omega and optimal weights are not in the same "position" of the network
            # TODO: Make EWC work better in the case of non-start of network shared weights...
            for omega, optimal, new in zip(self.omega_matrix[layer_index], self.optimal_weights[layer_index], layer.weights):
                loss += tf.reduce_sum(omega * tf.math.square(new-optimal))
        return loss * self.lam/2


class EWC_Term_Creator():

    def __init__(self, ewc_method: EWC_Method, model: tf.keras.models.Model, tasks: List[SequentialTask]) -> None:
        """
        Initialize a new creator for EWC terms

        Parameters:
            method: EWC_Method
                Enum to set term creation method, e.g. sign_flip, weight_decay

            model: tf.keras.models.Model
                The model to base EWC off of
        """
        self.ewc_method = ewc_method
        self.model = model
        self.model_layers = model.layers
        self.tasks=tasks
        self.callback_dict:dict[str, tf.keras.callbacks.Callback] = {}
        match self.ewc_method:
            case EWC_Method.SIGN_FLIPPING:
                self.callback_dict["SignFlip"] = SignFlippingTracker(model)
            case EWC_Method.FISHER_MATRIX:
                self.callback_dict["FisherCalc"] = FisherInformationMatrixCalculator(tasks)
            case _:
                pass

    def create_term(self, lam: float) -> EWC_Term:
        """
        Create a new term using whatever method is specified and whatever
        data is collected at the call time. Should only be called at the
        end of a task!

        Parameters:
            lam: float
                The importance of the new term
        """

        # Get current value of model weights
        # Also set a default value of omega matrix if useful
        # Default here is matrix of zeros
        model_current_weights = []
        omega_matrix = []
        for layer_index, layer in enumerate(self.model_layers):
            current_weights = []
            current_omega = []
            for weight_index, weight in enumerate(layer.weights):
                current_weights.append(weight)
                current_omega.append(tf.zeros_like(weight))
            model_current_weights.append(current_weights)
            omega_matrix.append(current_omega)

        match self.ewc_method:
            case EWC_Method.NONE:
                return EWC_Term(lam=0, optimal_weights=model_current_weights, omega_matrix=omega_matrix)
            
            case EWC_Method.WEIGHT_DECAY:
                # weight decay has omega as a matrix of 1's, which we can get from our already
                # calculated matrix of 0's !
                for layer_index, layer in enumerate(omega_matrix):
                    for weight_index, weight in enumerate(layer):
                        omega_matrix[layer_index][weight_index] += 1
                return EWC_Term(lam=lam,optimal_weights=model_current_weights, omega_matrix=omega_matrix)
            
            case EWC_Method.SIGN_FLIPPING:
                sign_flip_callback: SignFlippingTracker = self.callback_dict["SignFlip"]  # type: ignore
                omega_matrix = []
                for layer_index, layer in enumerate(sign_flip_callback.sign_changes):
                    omega_layer = []
                    for weight_index, weight in enumerate(layer):
                        # An important weight flips *fewer* times
                        # So we take the reciprocal, which could
                        # Divide by zero, so also add 1
                        omega_layer.append(1/(1+weight))
                    omega_matrix.append(omega_layer)
                return EWC_Term(lam=lam,optimal_weights=model_current_weights, omega_matrix=omega_matrix)
            
            # case EWC_Method.FISHER_MATRIX:
                # Calculate Fisher matrix and use as omega
                # To do this, need to run model over training dataset
                # And use this to approximate Fisher.
                #
                # Details here: https://towardsdatascience.com/an-intuitive-look-at-fisher-information-2720c40867d8
                # Example implementation: https://github.com/db434/EWC/blob/master/ewc.py
                # Original paper: https://arxiv.org/pdf/1612.00796.pdf

            # Default case: return an empty term
            case _:
                return EWC_Term(lam=0, optimal_weights=model_current_weights, omega_matrix=omega_matrix)


class EWC_Loss(tf.keras.losses.Loss):
    def __init__(self, base_loss: tf.keras.losses.Loss,
                 current_model_layers: List[tf.keras.layers.Layer],
                 EWC_terms: List[EWC_Term]):

        super().__init__()
        self.base_loss = base_loss
        self.model_layers = current_model_layers
        self.ewc_terms = EWC_terms

    def call(self, y_true, y_pred):
        base_loss = self.base_loss(y_true, y_pred)
        ewc_loss = 0
        for term in self.ewc_terms:
            ewc_loss += term.calculate_loss(self.model_layers)
        return base_loss + ewc_loss
