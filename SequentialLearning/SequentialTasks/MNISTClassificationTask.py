# fmt: off
from typing import List, Tuple
from MyUtils import normalize_img

from .SequentialTask import SequentialTask

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_datasets as tfds
# fmt: on

class MNISTClassificationTask(SequentialTask):
    """
    Create a new task based around classifying between different MNIST digit images
    The MNIST data is taken from tensorflow_datasets and is processed slightly to 
    improve performance

    Note for this task, the dataset originally has numeric labels (not one-hot)
    In the create dataset method we map these labels to one_hot to make modelling easier
    For example, a task that is classifying between the digits 3 and 4 does not need 10 outputs 
    It is recommended to use CategoricalLoss or BinaryCategoricalLoss
    """

    (full_training_dataset, full_validation_dataset), ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
    full_training_dataset: tf.data.Dataset = full_training_dataset.map(normalize_img,num_parallel_calls=tf.data.AUTOTUNE)
    full_validation_dataset: tf.data.Dataset = full_validation_dataset.map(normalize_img,num_parallel_calls=tf.data.AUTOTUNE)

    def __init__(self, 
            name: str,
            model: tf.keras.models.Model,
            model_base_loss: tf.keras.losses.Loss,
            task_digit_labels: List[int],
            training_batches: int = 0,
            validation_batches: int = 0,
            batch_size: int = 32,
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

            task_digit_labels: List[int]
                The digits to differentiate in this task
                Usually a list of two digits (e.g. [0,1]) for binary classification
                But can be larger (e.g. [0,1,2,3]) for a larger classification task
                Must be valid MNIST digits (0-9), list passed to dataset.filter

            training_batches: int
                The number of batches in the training dataset
                If 0 (default) use all batches available
            
            validation_batches: int
                The number of batches in the validation dataset
                If 0 (default) use all batches available

            batch_size: int
                The batch size for datasets
                Defaults to 128.

            **kwargs
                Other keyword arguments to be passed to super()
                Anything in this set is optional for this task 
                e.g. optional SequentialTask parameters
        """

        self.task_digit_labels = task_digit_labels
        self.training_batches = training_batches \
            if training_batches!=0 \
            else int(MNISTClassificationTask.ds_info.splits["train"].num_examples/batch_size)
        self.validation_batches = validation_batches \
            if validation_batches!=0 \
            else int(MNISTClassificationTask.ds_info.splits["test"].num_examples/batch_size)
        self.batch_size = batch_size

        (training_dataset, validation_dataset) = self.create_datasets()
        super().__init__(
            name = name,
            model = model,
            model_base_loss = model_base_loss,
            training_dataset=training_dataset,
            training_batches = training_batches,
            validation_dataset=validation_dataset,
            validation_batches = validation_batches,
            **kwargs)

    def create_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Creates (and returns) a tuple of (training_dataset, validation_dataset)
        based on the Iris dataset
        """

        filter_range = tf.constant(self.task_digit_labels, dtype=tf.int64)
        one_hot_depth = len(self.task_digit_labels)

        # We need to filter out only the labels we actually want before preprocessing
        training_samples = self.training_batches * self.batch_size
        training_dataset = MNISTClassificationTask.full_training_dataset \
            .filter(lambda _, label: tf.reduce_any(tf.equal(label, filter_range)))

        # This is an ugly hack to map numbers to cardinal values
        # e.g. if task digits are (7,4,5) this loop maps the results to (0,1,2)
        # ideally combine this loop and mapping to the map below (one-hot encoding) but... it works
        for final_val, init_val in enumerate(self.task_digit_labels):
            final_tensor = tf.constant(final_val, dtype=tf.int64)
            training_dataset = training_dataset.map(lambda x,y: (x, final_tensor if y==init_val else y))

        training_dataset = training_dataset \
            .map(lambda x,y: (x,tf.one_hot(y, depth=one_hot_depth))) \
            .take(training_samples) \
            .shuffle(training_samples) \
            .batch(self.batch_size) \
            .repeat() \
            .prefetch(tf.data.experimental.AUTOTUNE)

        # Repeat the same with validation
        validation_samples = self.validation_batches * self.batch_size
        validation_dataset = MNISTClassificationTask.full_validation_dataset \
            .filter(lambda _, label: tf.reduce_any(tf.equal(label, filter_range)))


        for final_val, init_val in enumerate(self.task_digit_labels):
            final_tensor = tf.constant(final_val, dtype=tf.int64)
            validation_dataset = validation_dataset.map(lambda x,y: (x, final_tensor if y==init_val else y))

        validation_dataset = validation_dataset \
            .map(lambda x,y: (x,tf.one_hot(y, depth=one_hot_depth))) \
            .take(validation_samples) \
            .shuffle(validation_samples) \
            .batch(self.batch_size) \
            .repeat() \
            .prefetch(tf.data.experimental.AUTOTUNE)
        
        return (training_dataset, validation_dataset)