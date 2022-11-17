# fmt: off
from copy import deepcopy
from typing import List, Union
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

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



