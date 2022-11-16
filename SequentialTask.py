# fmt: off
from typing import List, Tuple
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class SequentialTask:
    """
    A container for a single sequential task. 
    Includes a model (already compiled, hopefully sharing weights with other tasks)
    and a tf.data.Dataset representing the task data (with optional validation data)
    """

    def __init__(self, 
        name: str,
        model: tf.keras.models.Sequential,
        model_base_loss: tf.keras.losses.Loss,
        training_data: tf.data.Dataset,
        train_steps_per_epoch: int,
        validation_data: tf.data.Dataset = None,
        validation_steps_per_epoch: int = 0,
        input_data_fn = None,
        data_fn = None,
        x_lim:Tuple[float, float] = None,
        y_lim:Tuple[float, float] = None,
        ) -> None:
        """
        Create a new SequentialTask.
        A task consists of a model (already compiled), training data,
        and validation data to test the model. 

        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            model: tf.keras.models.Sequential
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)

            training_data: tf.data.Dataset
                The training data to fit to

            train_steps_per_epoch: int
                The number of batches in the training dataset

            validation_data: tf.data.Dataset
                The validation data to test on. Optional, if None no validation is done

            validation_steps_per_epoch: int
                The number of batches in the validation dataset

            input_data_fn: function
                The function used to create this task input data (single independancy only)

            data_fn: function
                The function used to map inputs to outputs (if applicable)
                NOT only single independant variable

            x_lim: Tuple[float, float]
                The input limits of this task, if applicable (single independant variable only)

            y_lim: Tuple[float, float]
                The output limits of the task, for single outputs only
        """

        self.name = name
        self.model = model
        self.model_base_loss = model_base_loss
        self.training_data = training_data
        self.training_steps_per_epoch = train_steps_per_epoch
        self.validation_data = validation_data
        self.validation_steps_per_epoch = validation_steps_per_epoch
        self.input_data_fn = input_data_fn
        self.data_fn = data_fn
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.compile_model(model_base_loss)


    def compile_model(self, loss_fn: tf.keras.losses.Loss):
        """
        (Re)compile this tasks model with a new loss function, keeping the metrics
        """

        self.model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=[self.model_base_loss])

    def train_on_task(self, epochs, callbacks: List[tf.keras.callbacks.Callback]) -> tf.keras.callbacks.History:
        """
        Train on the train dataset for a number of epochs. Use any callbacks given
        If self.validation_data is not None, validation data used.
        Returns the history of training
        """

        return self.model.fit(
            self.training_data,
            epochs=epochs,
            steps_per_epoch=self.training_steps_per_epoch,
            validation_data=self.validation_data,
            validation_steps=self.validation_steps_per_epoch,
            callbacks=callbacks
        )

    def evaluate_model(self):
        """
        Run a single pass over the validation data, returning the metrics
        """

        if self.validation_data is None:
            return 0
            
        return self.model.evaluate(self.validation_data, 
            steps=self.validation_steps_per_epoch, return_dict=True)