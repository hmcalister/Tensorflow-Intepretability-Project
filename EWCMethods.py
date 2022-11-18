# fmt: off
from copy import deepcopy
from typing import List, Union
import numpy as np
import os
from SequentialTask import SequentialTask
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

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
        # For application, that means base_model (with, say, 2 layers) has 
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

class EWC_Term():
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
            for omega, optimal, new in zip(self.omega_matrix[layer_index], self.optimal_weights[layer_index], layer.weights):
                loss += tf.reduce_sum(omega * tf.math.square(new-optimal))
        return loss * self.lam/2

class EWC_Loss(tf.keras.losses.Loss):
    def __init__(self, base_loss: tf.keras.losses.Loss, 
        current_model_layers: List[tf.keras.layers.Layer],
        EWC_terms: List[EWC_Term]):

        super().__init__()
        self.base_loss = base_loss
        self.model_layers = current_model_layers
        self.ewc_terms = EWC_terms

    def call(self, y_true, y_pred):
        loss = self.base_loss(y_true, y_pred)
        for term in self.ewc_terms:
            loss += term.calculate_loss(self.model_layers)
        return loss



