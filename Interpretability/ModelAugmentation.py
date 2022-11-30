# fmt: off
from copy import deepcopy
from enum import Enum
from typing import List
import numpy as np

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class ComparisonMethod(Enum):
    LAYER_WISE = 0,
    MODEL_WISE = 1,

def threshold_by_omega(
    base_model: tf.keras.models.Model,
    omega_matrix: List[List[np.ndarray]],
    threshold_percentage: float,
    comparison_method: ComparisonMethod
    ) -> tf.keras.models.Model:
    """
    Given a base model and some measure of weight importance, create a new model
    that has the same weights as the base model for the most important weights and
    zero for non-important weights

    Implementation first finds the distribution of weights, then determines which 
    weights are above the threshold value supplied (as a proportion) and zeros weights
    with importance below the threshold

    Parameters:
        base_model: tf.keras.models.Model
            The model to augment, taking weights from
            The model is copied, so the original model is unaffected by this operation
        omega_matrix: List[List[np.ndarray]]
            The measure of weight importance to apply to the base model
            Note this requires the shapes of this matrix and model weights be the same
        threshold_percentage: float
            The threshold of importance to keep an associated weight
            Given as a float between 0 (keep all weights) and 1 (keep no weights)
            Note this is percentage i.e. linear. Not a standard deviation or Z score
        comparison_method: ComparisonMethod
            Method for determining actual threshold values i.e. compare weights 
            only within a layer (LAYER_WISE) or across the entire model (MODEL_WISE)
    """

    new_model = tf.keras.models.clone_model(base_model)
    threshold_value: np.float32 = np.float32(0)
    
    # If we are comparing across the entire model, do this before thresholds
    if comparison_method == ComparisonMethod.MODEL_WISE:
        flat_omega = []  # type: ignore
        for layer_index, layer in enumerate(omega_matrix):
            for weight_index, omega in enumerate(layer):
                flat_omega = tf.concat([flat_omega, tf.reshape(omega, [-1])], axis=0)
        flat_omega = tf.sort(flat_omega)
        threshold_index = int(len(flat_omega) * threshold_percentage)
        threshold_value = flat_omega[threshold_index].numpy()
        print(f"MODEL_WISE {threshold_value=}")

    for layer_index, (layer_omega, model_layer) in enumerate(zip(omega_matrix, base_model.layers)):
        if comparison_method == ComparisonMethod.LAYER_WISE:
            flat_omega = []  # type: ignore
            for weight_index, (omega, _) in enumerate(zip(layer_omega, model_layer.weights)):
                flat_omega = tf.concat([flat_omega, tf.reshape(omega, [-1])], axis=0)
            flat_omega = tf.sort(flat_omega)
            if len(flat_omega)==0:
                # This is a strange condition
                # Effectively, Input layers have NO weights, nada, none, []
                # So we cannot even index into them with threshold_index = 0
                # instead we just hack a value to prevent a crash and move on
                threshold_value = np.float32(0)
            else:
                threshold_index = int(len(flat_omega) * threshold_percentage)
                threshold_value = flat_omega[threshold_index].numpy()
                print(f"LAYER_WISE {layer_index=} {threshold_value=}")

        new_layer_weights = []
        for weight_index, (omega, weight) in enumerate(zip(layer_omega, model_layer.weights)):
            new_weight = deepcopy(weight.numpy())
            new_weight[omega < threshold_value] = 0
            new_layer_weights.append(new_weight)
        new_model.layers[layer_index].set_weights(new_layer_weights)

    # Yeah, I'm setting a private field, sue me
    new_model._name = f"{threshold_percentage}-threshold_{base_model.name}"
    optimizer = deepcopy(base_model.optimizer)
    loss_fn = deepcopy(base_model.loss)
    run_eagerly = base_model.run_eagerly
    new_model.compile(
        optimizer=optimizer,
        loss = loss_fn,
        run_eagerly=run_eagerly
    )

    return new_model