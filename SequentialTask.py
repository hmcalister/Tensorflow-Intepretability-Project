# fmt: off
from typing import Callable, List, Tuple, Union
import numpy as np
import os
import pandas as pd

from MyUtils import normalize_img
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds
# fmt: on

RUN_EAGERLY=True

class SequentialTask:
    """
    A container for a single sequential task. 
    Includes a model (already compiled, hopefully sharing weights with other tasks)
    and a tf.data.Dataset representing the task data (with optional validation data)
    """

    def __init__(self, 
            name: str,
            model: tf.keras.models.Model,
            model_base_loss: tf.keras.losses.Loss,
            training_dataset: tf.data.Dataset,
            training_batches: int,
            validation_dataset: Union[tf.data.Dataset, None] = None,
            validation_batches: int = 0,
            batch_size: int = 0,
            input_data_fn: Union[Callable, None] = None,
            data_fn: Union[Callable, None] = None,
            x_lim: Union[Tuple[float, float], None] = None,
            y_lim: Union[Tuple[float, float], None] = None,
        ) -> None:
        """
        Create a new SequentialTask.
        A task consists of a model (already compiled), training data,
        and validation data to test the model. 

        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            model: tf.keras.models.Model
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)

            training_data: tf.data.Dataset
                The training data to fit to

            training_batches: int
                The number of batches in the training dataset

            validation_data: tf.data.Dataset
                The validation data to test on. Optional, if None no validation is done

            validation_batches: int
                The number of batches in the validation dataset

            input_data_fn: function
                The function used to create this task input data (single independency only)

            data_fn: function
                The function used to map inputs to outputs (if applicable)
                NOT only single independent variable

            x_lim: Tuple[float, float]
                The input limits of this task, if applicable (single independent variable only)

            y_lim: Tuple[float, float]
                The output limits of the task, for single outputs only
        """

        self.name = name
        self.model = model
        self.model_base_loss = model_base_loss
        self.training_dataset = training_dataset
        self.training_batches = training_batches
        self.validation_dataset = validation_dataset
        self.validation_batches = validation_batches
        self.batch_size = batch_size
        self.input_data_fn = input_data_fn
        self.data_fn = data_fn
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.compile_model(model_base_loss)


    def compile_model(self, loss_fn: tf.keras.losses.Loss):
        """
        (Re)compile this tasks model with a new loss function, keeping the metrics
        """

        # Notice that EWC requires access to layer weights during training
        # If using numpy arrays (easier debugging)/layer.get_weights()
        # this is not possible with a compiled graph (Tensorflow restricts it)
        # So we must set run_eagerly to True to avoid compilation, or
        # we can use .weights and use tensorflow Tensors instead!
        self.model.compile(optimizer='ADAM',
                loss=loss_fn,
                metrics=[self.model_base_loss],
                run_eagerly=RUN_EAGERLY)

    def train_on_task(self, epochs, callbacks: List[tf.keras.callbacks.Callback]) -> tf.keras.callbacks.History:
        """
        Train on the train dataset for a number of epochs. Use any callbacks given
        If self.validation_data is not None, validation data used.
        Returns the history of training
        """

        return self.model.fit(
            self.training_dataset,
            epochs=epochs,
            steps_per_epoch=self.training_batches,
            validation_data=self.validation_dataset,
            validation_steps=self.validation_batches,
            callbacks=callbacks
        )

    def evaluate_model(self) -> dict:
        """
        Run a single pass over the validation data, returning the metrics
        """

        if self.validation_dataset is None:
            return {}
            
        # Return type of this is hinted incorrectly
        # Actual return type is dict
        print(f"EVALUATING: {self.model.name}")
        return self.model.evaluate(self.validation_dataset, 
            steps=self.validation_batches, return_dict=True)  # type: ignore


class FunctionApproximationTask(SequentialTask):
    """
    A task about modelling a function that maps inputs (independent variables) to outputs
    Functions are the data functions, and (if independent variables != model_input_shape)
    input_data_function maps independent variables to input tensors
    """

    def __init__(self, 
            name: str, 
            model: tf.keras.models.Model, 
            model_base_loss: tf.keras.losses.Loss,
            independent_variables: int,
            model_input_shape: Tuple[int,],
            input_data_fn: Callable,
            data_fn: Callable,
            training_batches: int = 0,
            validation_batches: int = 0,
            batch_size: int = 32,
            x_lim: Tuple[float, float] = (-1,1),
            **kwargs,
        ) -> None:
        """
        Create a new FunctionApproximationTask.

        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            model: tf.keras.models.Model
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)
            
            independent_variables: int
                The number of independent variables to create for each data

            model_input_shape: Tuple[int,]
                The input shape of the model

            input_data_fn: function
                The function to map independent variables to model inputs
                (If no mapping required, use lambda x: x)

            data_fn: function
                The function to map independent variables to model outputs

            training_batches: int
                The number of batches in the training dataset
            
            validation_batches: int
                The number of batches in the validation dataset

            batch_size: int
                The batch size for datasets
                Defaults to 32.

            x_lim: Tuple[float, float]:
                The input limits to the data function
                Defaults to (-1,1)

            **kwargs
                Other keyword arguments to be passed to super()
                Anything in this set is optional for this task 
                e.g. optional SequentialTask parameters
        """

        self.model_input_shape = model_input_shape
        self.independent_variables = independent_variables
        self.batch_size = batch_size
        self.training_samples = batch_size
        self.training_samples = training_batches * batch_size
        self.validation_samples = validation_batches * batch_size

        super().__init__(
            name = name,
            model = model,
            model_base_loss = model_base_loss,
            input_data_fn = input_data_fn,
            data_fn = data_fn,
            training_dataset=self.create_dataset(training_batches * batch_size),
            training_batches = training_batches,
            validation_dataset=self.create_dataset(validation_batches * batch_size),
            validation_batches = validation_batches,
            x_lim = x_lim,
            **kwargs)

        # Set typing correctly
        self.x_lim: Tuple[float, float]
        self.input_data_fn: Callable
        self.data_fn: Callable

    def data_generator(self, max_samples):
        i = 0
        while i < max_samples:
            x = np.random.uniform(self.x_lim[0], self.x_lim[1], self.independent_variables)
            y = self.data_fn(x)
            yield self.input_data_fn(x), y
            i += 1

    def create_dataset(self, max_samples):
        return tf.data.Dataset.from_generator(
            self.data_generator,
            args=[max_samples],
            output_signature=(
                tf.TensorSpec(shape=self.model_input_shape, dtype=tf.float64),  # type: ignore
                tf.TensorSpec(shape=(), dtype=tf.float64),  # type: ignore
            )).batch(self.batch_size).repeat()

class IrisClassificationTask(SequentialTask):
    """
    Create a new classification task based on the Iris dataset
    Task consists of some subset of feature columns
    (['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
    being mapped to one-hot encoded label columns
    Loss should be categorical cross-entropy or similar for classification

    Warning! This dataset has only 150 items!
    Suggested training items is 120 and validation 30
    (Note: default batch_size=10, so choose training_batches=12, validation=3)
    """
    def __init__(self, 
            name: str,
            model: tf.keras.models.Model,
            model_base_loss: tf.keras.losses.Loss,
            feature_column_names: List[str],
            training_batches: int = 12,
            validation_batches: int = 3,
            batch_size: int = 10,
            iris_dataset_csv_path: str = "datasets/iris_csv.csv",
            **kwargs,
        ) -> None:
        """
        Create a new FunctionApproximationTask.

        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            model: tf.keras.models.Model
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)

            feature_column_names: List[str]
                The column names to be used as features in this task
                Must be valid column names for the iris dataset
                e.g. 
                    'sepallength', 
                    'sepalwidth', 
                    'petallength', 
                    'petalwidth'

            training_batches: int
                The number of batches in the training dataset
            
            validation_batches: int
                The number of batches in the validation dataset

            batch_size: int
                The batch size for datasets
                Defaults to 10.

            iris_dataset_csv_path: str
                String path to the iris dataset csv file

            **kwargs
                Other keyword arguments to be passed to super()
                Anything in this set is optional for this task 
                e.g. optional SequentialTask parameters
        """

        self.feature_column_names = feature_column_names
        self.original_dataframe = pd.read_csv(iris_dataset_csv_path)
        self.training_batches = training_batches
        self.validation_batches = validation_batches
        self.batch_size = batch_size

        (train_dataset, validation_dataset) = self.create_datasets()
        super().__init__(
            name = name,
            model = model,
            model_base_loss = model_base_loss,
            training_dataset=train_dataset,
            training_batches = training_batches,
            validation_dataset=validation_dataset,
            validation_batches = validation_batches,
            **kwargs)

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Creates (and returns) a tuple of (training_dataset, validation_dataset)
        based on the Iris dataset
        """

        # Here we define how many items we need from the training samples
        # Then group the feature columns by class (to evenly sample)
        # Before finally taking exactly the indices corresponding to
        # (0,1,...,training_samples) using np.arange
        # Note we divide by 3 because we know there are 3 classes
        training_samples = self.training_batches * self.batch_size
        training_dataframe = self.original_dataframe \
            .groupby("class") \
            .apply(lambda x: x.take(
                np.arange(training_samples/3)  # type: ignore
            ))
        # Repeat the same with validation but this time offset indices by
        # already consumed training_samples
        validation_samples = self.validation_batches * self.batch_size
        validation_dataframe = self.original_dataframe \
            .groupby("class") \
            .apply(lambda x: x.take(
                np.arange(training_samples/3, (training_samples+validation_samples)/3)  # type: ignore
            ))

        # Now each dataframe consists of unique items of evenly sampled classes,
        # We can process into tensorflow datasets

        training_features = training_dataframe[self.feature_column_names]
        training_labels = pd.get_dummies(training_dataframe["class"], prefix="class")
        training_dataset = tf.data.Dataset.from_tensor_slices((training_features,training_labels)) \
            .shuffle(training_samples) \
            .batch(self.batch_size)

        validation_features = validation_dataframe[self.feature_column_names]
        validation_labels = pd.get_dummies(validation_dataframe["class"], prefix="class")
        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_features, validation_labels)) \
            .shuffle(validation_samples) \
            .batch(self.batch_size)
        
        return (training_dataset, validation_dataset)
        return (training_dataset, validation_dataset)