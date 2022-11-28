# fmt: off
from MyUtils import *
from SequentialLearning.SequentialLearningManager import SequentialLearningManager
from SequentialLearning.SequentialTasks.MNISTClassificationTask import MNISTClassificationTask
from SequentialLearning.EWC_Methods.EWC_Methods import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

print(f"GPU: {tf.config.list_physical_devices('GPU')}")
model_input_shape = (28,28,1)

# Digits to classify in each task
# Should be taken from 0-9
task_digit_classes = [
    [0,1],
    [2,3],
    [4,5]
]

# base model for sequential tasks
# each model gets these layers as a base, then adds own head layers
# i.e. these weights are *shared*
base_model_inputs = base_model_layer = tf.keras.Input(shape=model_input_shape)
base_model_layer = tf.keras.layers.Conv2D(10, (5,5), activation="relu")(base_model_layer)
base_model_layer = tf.keras.layers.MaxPool2D((3,3))(base_model_layer)
base_model_layer = tf.keras.layers.Conv2D(10, (3,3), activation="relu")(base_model_layer)
base_model_layer = tf.keras.layers.Flatten()(base_model_layer)
base_model_layer = tf.keras.layers.Dense(10, activation="relu")(base_model_layer)
base_model = tf.keras.Model(inputs=base_model_inputs, outputs=base_model_layer, name="base_model")

# Layers specific to each task
# Not shared
task_head_layers = [
    [
        tf.keras.layers.Dense(2, activation="softmax")
    ],
    [
        tf.keras.layers.Dense(2, activation="softmax")
    ],
    [
        tf.keras.layers.Dense(2, activation="softmax")
    ],
]

# The base loss function for tasks
# Currently all tasks have the same structure so only one loss
# Could use a list in future
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Training parameters
epochs = 50
training_batches = 300
validation_batches = 50
batch_size = 32
ewc_method = EWC_Method.MOMENTUM_BASED

print(f"BASE MODEL SUMMARY")
base_model.summary()

# -----------------------------------------------------------------------------
# AUTOMATED SETUP: DON'T TOUCH BELOW HERE UNLESS CONFIDENT
# -----------------------------------------------------------------------------

# Create, compile, and build all models
models = []
for task_index in range(len(task_digit_classes)):
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
for task_index in range(len(task_digit_classes)):
    tasks.append(MNISTClassificationTask(
        name=f"Task {task_index+1}",
        model=models[task_index],
        model_base_loss=loss_fn,
        task_digit_labels=task_digit_classes[task_index],
        training_batches = training_batches,
        validation_batches = validation_batches,
        batch_size=batch_size
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
