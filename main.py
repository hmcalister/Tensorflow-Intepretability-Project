# fmt: off
import numpy as np
import matplotlib.pyplot as plt
from MyUtils import *
from MyCallbacks import *
import os
from SequentialLearningManager import SequentialLearningManager
from SequentialTask import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

# Inputs to network as well as input / output limits
# Note limits are not really used/enforced anywhere (outside animation graphing)
# But are useful for documenting expected behavior
independent_variables = 5
model_input_shape = (5,)
x_lim = (-1,1)
y_lim = None

# Function to define how to map independent variables to model inputs
input_data_fn = lambda x: x

# Functions to model in sequential tasks
# Notice that for a data_fn like lambda x:...
# x is a *list* of inputs! Even if that list is one element
data_fns = [
    lambda x: np.sum(x),
    lambda x: np.sum(np.sin(3*x))
]

# base model for sequential tasks
# each model gets these layers as a base, then adds own head layers
# i.e. these weights are *shared*
base_model_inputs = tf.keras.Input(shape=model_input_shape)
base_model_layer = tf.keras.layers.Dense(10, activation="relu")(base_model_inputs)
base_model = tf.keras.Model(inputs=base_model_inputs, outputs=base_model_layer, name="base_model")

# Layers specific to each task
# Not shared
task_head_layers = [
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
loss_fn.name = "base_loss"

# Training parameters
epochs = 5
batch_size = 64
training_batches = 128
validation_batches = 32
items_per_epoch = batch_size * training_batches

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
    
    curr_model = tf.keras.Model(inputs=base_model_inputs, outputs=curr_model_layer, name=f"task_{task_index}_model")
    models.append(curr_model)

# Create the task representations (see SequentialTask)
tasks = []
for task_index in range(len(data_fns)):
    tasks.append(FunctionApproximationTask(
        name=f"Task {task_index+1}",
        model=models[task_index],
        model_base_loss=loss_fn,
        independent_variables=independent_variables,
        model_input_shape=model_input_shape,
        input_data_fn = input_data_fn,
        data_fn = data_fns[task_index],
        training_batches = training_batches,
        validation_batches = validation_batches,
        batch_size=batch_size,
        x_lim = x_lim,
        y_lim = y_lim,
    ))


# Create the manager
manager = SequentialLearningManager(base_model, tasks, epochs)
# Train all tasks sequentially
manager.train_all()
# Plot output data
manager.plot_validation_callback_data("loss", title="Task EWC Losses Over All Epochs", ylabel="EWC Loss")
manager.plot_validation_callback_data("base_loss", title="Task Base Losses Over All Epochs", ylabel="Base Loss")
manager.plot_task_training_histories()