# fmt: off
import numpy as np
import matplotlib.pyplot as plt
from Utilities.Utils import *
from Utilities.SequentialLearning.SequentialLearningManager import SequentialLearningManager
from Utilities.SequentialLearning.Tasks.IrisClassificationTask import IrisClassificationTask as Task
from Utilities.SequentialLearning.EWC_Methods.EWC_Methods import EWC_Method

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

model_input_shape = (2,)

# Feature names of iris dataset
# Each new list will produce a new task with those labels as features
# Possible labels are ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
task_feature_names = [
    ['sepallength', 'petallength'],
    ['petalwidth', 'sepallength'],
    ['sepalwidth', 'petalwidth'],
]

# base model for sequential tasks
# each model gets these layers as a base, then adds own head layers
# i.e. these weights are *shared*
base_model_inputs = base_model_layer = tf.keras.Input(shape=model_input_shape)
base_model_layer = tf.keras.layers.Dense(10, activation="relu")(base_model_layer)
base_model_layer = tf.keras.layers.Dense(10, activation="relu")(base_model_layer)
base_model_layer = tf.keras.layers.Dense(10, activation="relu")(base_model_layer)
# base_model_layer = tf.keras.layers.Dense(1)(base_model_layer)
base_model = tf.keras.Model(inputs=base_model_inputs, outputs=base_model_layer, name="base_model")

# Layers specific to each task
# Not shared
task_head_layers = [
    [
        tf.keras.layers.Dense(3, activation="softmax")
    ],
    [
        tf.keras.layers.Dense(3, activation="softmax")
    ],
    [
        tf.keras.layers.Dense(3, activation="softmax")
    ],
]

# The base loss function for tasks
# Currently all tasks have the same structure so only one loss
# Could use a list in future
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
loss_fn.name = "base_loss"

# Training parameters
epochs = 10
ewc_method = EWC_Method.FISHER_MATRIX


print(f"BASE MODEL SUMMARY")
base_model.summary()

# -----------------------------------------------------------------------------
# AUTOMATED SETUP: DON'T TOUCH BELOW HERE UNLESS CONFIDENT
# -----------------------------------------------------------------------------

# Create, compile, and build all models
models = []
for task_index in range(len(task_feature_names)):
    if task_head_layers is None:
        layers = []
    else:
        layers = task_head_layers[task_index]

    curr_model_layer = base_model_layer
    for layer in layers:
        curr_model_layer = layer(curr_model_layer)

    curr_model = tf.keras.Model(
        inputs=base_model_inputs, outputs=curr_model_layer, name=f"task_{task_index+1}_model")
    models.append(curr_model)

# Create the task representations (see SequentialTask)
tasks = []
for task_index in range(len(task_feature_names)):
    tasks.append(Task(
        name=f"Task {task_index+1}",
        model=models[task_index],
        model_base_loss=loss_fn,
        feature_column_names=task_feature_names[task_index],
    ))


# Create the manager
manager = SequentialLearningManager(base_model, tasks, epochs, ewc_method)
# Train all tasks sequentially
manager.train_all()
# Plot output data
manager.plot_validation_callback_data(
    "loss", title="Task Total Loss Over All Epochs", ylabel="Total Loss (CategoricalCrossentropy)")
manager.plot_validation_callback_data(
    "base_loss", title="Task Base Losses Over All Epochs", ylabel="Base Loss (CategoricalCrossentropy)")
multiplot_data(manager.get_training_histories())
