# fmt: off
from copy import deepcopy
from typing import List, Union
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class EWCTerm():
    def __init__(self,
        lam: float,
        optimal_weights: List[np.ndarray], 
        omega_matrix:List[np.ndarray]):
        """
        A single EWC term for model training

        Parameters:
            lam: float
                The importance of this EWC term. 

            optimal_weights: List[np.ndarray]
                The optimal weights of the model after training.
                Can be found by model.get_weights()
                Note! Should only be the *shared* weights 
            
            omega_matrix: List[np.ndarray]
                The weight importance matrix for this term.
                Should have the same dimensions (in every way) as 
                optimal_weights
        """

        self.optimal_weights = optimal_weights
        self.omega_matrix = omega_matrix

    def create_loss(self):
        """
        Create (and return) a new loss function based on elastic weight consolidation
        At each call, finds the squared difference of optimal_weights and new weights, multiplied by omega
        """

        lam = self.lam
        optimal_weights = deepcopy(self.optimal_weights)
        omega_matrix = deepcopy(self.omega_matrix)


        def loss_fn(model: tf.keras.models.Sequential):
            loss = 0
            new_weights = model.weights
            for omega, optimal, new in zip(omega_matrix, optimal_weights, new_weights):
                loss += tf.reduce_sum(omega * tf.math.square(new-optimal))

            return loss * lam/2

        return loss_fn