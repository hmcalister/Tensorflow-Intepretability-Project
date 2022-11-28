# fmt: off
from copy import deepcopy
import os
from typing import List

from SequentialLearning.SequentialTask import SequentialTask
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

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
        for x, _ in current_task.training_dataset.take(samples):
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
