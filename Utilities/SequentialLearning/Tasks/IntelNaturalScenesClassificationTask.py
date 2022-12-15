# fmt: off
from typing import List, Tuple
import numpy as np
import pandas as pd

from .GenericTask import GenericTask

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

class IntelNaturalScenesClassificationTask(GenericTask):
    """
    Create a new IntelNaturalScenes Classification Task
    This dataset is based around classifying scenes based on images
    Interestingly these images are large and non-uniform

    https://www.kaggle.com/datasets/puneet6060/intel-image-classification
    """

    def __init__(self, 
            name: str,
            model: tf.keras.models.Model,
            model_base_loss: tf.keras.losses.Loss,
            feature_column_names: List[str],
            batch_size: int = 32,
            training_batches: int = 0,
            validation_batches: int = 0,
            training_image_augmentation: tf.keras.Sequential | None = None,
            data_path: str = "datasets/IntelNaturalScenes",
            **kwargs,
        ) -> None:
        """
        Create a new IntelNaturalScenes classification task.

        Parameters:
            name: str
                The name of this task. Usually like "Task 1"

            model: tf.keras.models.Model
                The model to fit to the tasks data

            model_base_loss: tf.keras.losses.Loss:
                The base loss function of the model (before EWC)
            
            feature_column_names: List[str]
                The names of features to use for this task
                Can be from the list:
                    "buildings"
                    "forest"
                    "glacier"
                    "mountain"
                    "sea"
                    "street"
            
            batch_size: int
                The batch size for training

            training_batches: int
                The number of batches to take for training
                If 0 (default) use as many batches as possible

            validation_batches: int
                The number of batches to take for validation
                If 0 (default) use as many batches as possible

            training_image_augmentation: tf.keras.Sequential | None
                A pipeline to augment the training images before training
                e.g. add random resizing, zooming, ...
                See https://pyimagesearch.com/2021/06/28/data-augmentation-with-tf-data-and-tensorflow/
                If None no augmentation is applied

            data_path: str
                The path to the IntelNaturalScenes Dataset
                Should contain three unzipped folders
                seg_train, seg_test, seg_pred
            """

        self.feature_column_names = feature_column_names
        self.batch_size = batch_size
        self.training_batches = training_batches 
        self.validation_batches = validation_batches
        self.training_image_augmentation: tf.keras.Sequential = training_image_augmentation
        self.data_path = data_path


    def _load_data(self):
        """
        Loads the data from the specified data path
        If any augmentation specified apply this too
        """

        