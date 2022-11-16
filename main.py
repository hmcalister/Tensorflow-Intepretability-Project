# fmt: off
import numpy as np
import matplotlib.pyplot as plt
from MyUtils import *
from MyCallbacks import *
import os
from SequentialLearningManager import SequentialLearningManager
from SequentialTask import SequentialTask

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

# Inputs to network as well as input / output limits
# Note limits are not really used/enforced anywhere (outside animation graphing)
# But are useful for documenting expected behavior
independant_variables = 1
model_input_shape = (5,)
input_lim = (-1, 1)
y_lim = (-1,1)

# Functions to model in sequential tasks
# Notice that for a data_fn like lambda x:...
# x is a *list* of inputs! Even if that list is one element
data_fns = [
    lambda x: np.sin(3*x[0]),
    lambda x: np.cos(3*x[0])
]

# base model for sequential tasks
# each model gets these layers as a base, then adds own head layers
# i.e. these weights are *shared*
base_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=model_input_shape),
    tf.keras.layers.Dense(10, activation="relu"),
])

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
epochs = 10
batch_size = 32
train_batches = 256
validation_batches = 16
items_per_epoch = batch_size * train_batches

# Some basic constants to be used throughout the script
def powers_set(x): return np.array([x[0]**i for i in range(1, model_input_shape[0]+1)])

# Function to define how to map independant variables to model inputs
input_data_fn = powers_set


# -----------------------------------------------------------------------------
# AUTOMATED SETUP: DON'T TOUCH BELOW HERE UNLESS CONFIDENT
# -----------------------------------------------------------------------------

def data_generator(task_index, max_samples):
    i = 0
    while i < max_samples:
        x = np.random.uniform(input_lim[0], input_lim[1], independant_variables)
        y = data_fns[task_index](x)
        # Return a number of powers of x
        yield powers_set(x), y
        i += 1

# Create, compile, and build all models
models = []
for task_index in range(len(data_fns)):
    if task_index >= len(task_head_layers):
        layers = []
    else:
        layers = task_head_layers[task_index]

    # Use splat operator to expand list into sequential arguement
    model = tf.keras.models.Sequential([
        base_model,
        *layers
    ])
    
    models.append(model)


def create_train_dataset(task_index):
    return tf.data.Dataset.from_generator(
        data_generator,
        args=[task_index, batch_size*train_batches],
        output_signature=(
            tf.TensorSpec(shape=model_input_shape, dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        )).batch(batch_size).repeat()


def create_validation_dataset(task_index):
    return tf.data.Dataset.from_generator(
        data_generator,
        args=[task_index, batch_size*validation_batches],
        output_signature=(
            tf.TensorSpec(shape=model_input_shape, dtype=tf.float64),
            tf.TensorSpec(shape=(), dtype=tf.float64),
        )).batch(batch_size).repeat()


# Create the task representations (see SequentialTask)
tasks = []
for task_index in range(len(data_fns)):
    # Create a test and validation dataset for the task (notice reliance on task index allows different datasets)
    train_dataset = create_train_dataset(task_index)
    validation_dataset = create_validation_dataset(task_index)

    tasks.append(SequentialTask(
        name=f"Task {task_index+1}",
        model=models[task_index],
        model_base_loss=loss_fn,
        training_data = train_dataset,
        train_steps_per_epoch = train_batches,
        validation_data = validation_dataset,
        validation_steps_per_epoch = validation_batches,
        input_data_fn = input_data_fn,
        data_fn = data_fns[task_index],
        x_lim = input_lim,
        y_lim = y_lim
    ))


# Create the manager
manager = SequentialLearningManager(base_model, tasks, epochs)
# Train all tasks sequentially
manager.train_all()
# Plot output data
manager.plot_validation_callback_data("loss", title="Task EWC Losses Over All Epochs", ylabel="EWC Loss")
manager.plot_validation_callback_data("base_loss", title="Task Base Losses Over All Epochs", ylabel="Base Loss")
manager.plot_task_training_histories()