# fmt: off
import numpy as np
from Utilities.Utils import *
from Utilities.SequentialLearning.SequentialLearningManager import SequentialLearningManager
from Utilities.Tasks.FunctionApproximationTask import FunctionApproximationTask as Task
from Utilities.SequentialLearning.EWC_Methods.EWC_Methods import EWC_Method

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

# Inputs to network as well as input / output limits
independent_variables = 5
model_input_shape = (5,)
x_lim = (-1, 1)

# Function to define how to map independent variables to model inputs
# Notice that for a input_data_fn like lambda x:...
input_data_fn = lambda x: x

# Functions to model in sequential tasks
# Notice that for a data_fn like lambda x:...
# x is a *list* of inputs! Even if that list is one element
data_fns = [
    # Alternating sum of elements
    lambda x: np.sum([(-1**i) * n for i,n in enumerate(x)]),
    # Sum of different trig functions
    lambda x: np.sum([np.sin(n + i*np.pi/4) for i,n in enumerate(x)]),
    # sum of sinc's
    lambda x: np.sum(np.sinc(x))
]

# base model for sequential tasks
# each model gets these layers as a base, then adds own head layers
# i.e. these weights are *shared*
base_model_inputs = base_model_layer = tf.keras.Input(shape=model_input_shape)
base_model_layer = tf.keras.layers.Dense(64, activation="relu")(base_model_layer)
base_model_layer = tf.keras.layers.Dense(32, activation="relu")(base_model_layer)
base_model_layer = tf.keras.layers.Dense(32, activation="relu")(base_model_layer)
base_model = tf.keras.Model(inputs=base_model_inputs, outputs=base_model_layer, name="base_model")

# Layers specific to each task
# Not shared
task_head_layers = [
    [
        tf.keras.layers.Dense(1)
    ],
    [
        tf.keras.layers.Dense(1)
    ],
    [
        tf.keras.layers.Dense(1)
    ]
]

# The base loss function for tasks
# Currently all tasks have the same structure so only one loss
# Could use a list in future
loss_fn = tf.keras.losses.MeanSquaredError()

# Training parameters
epochs = 10
batch_size = 64
training_batches = 256
validation_batches = 64
items_per_epoch = batch_size * training_batches
ewc_method = EWC_Method.NONE

print(f"BASE MODEL SUMMARY")
base_model.summary()

# -----------------------------------------------------------------------------
# AUTOMATED SETUP: DON'T TOUCH BELOW HERE UNLESS CONFIDENT
# -----------------------------------------------------------------------------

# Create, compile, and build all models
models = []
for task_index in range(len(data_fns)):
    if task_index >= len(task_head_layers):
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
for task_index in range(len(data_fns)):
    tasks.append(Task(
        name=f"Task {task_index+1}",
        model=models[task_index],
        model_base_loss=loss_fn,
        independent_variables=independent_variables,
        model_input_shape=model_input_shape,
        input_data_fn=input_data_fn,
        data_fn=data_fns[task_index],
        training_batches=training_batches,
        validation_batches=validation_batches,
        batch_size=batch_size,
        x_lim=x_lim,
    ))


# Create the manager
manager = SequentialLearningManager(base_model, tasks, epochs, ewc_method)
# Train all tasks sequentially
manager.train_all()
# Plot output data
manager.plot_validation_callback_data(
    "loss", title="Task Total Loss Over All Epochs", ylabel="Total Loss")
manager.plot_validation_callback_data(
    "base_loss", title="Task Base Losses Over All Epochs", ylabel="Base Loss")
multiplot_data(manager.get_training_histories())