# fmt: off
from Utilities.Utils import *
from Utilities.SequentialLearning.SequentialLearningManager import SequentialLearningManager
from Utilities.SequentialLearning.Tasks.IntelNaturalScenesClassificationTask import IntelNaturalScenesClassificationTask as Task
from Utilities.SequentialLearning.EWC_Methods.EWC_Methods import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

# True for easier debugging
# False for compiled models, faster train time
RUN_EAGERLY: bool = False

print(f"GPU: {tf.config.list_physical_devices('GPU')}")
model_input_shape = Task.IMAGE_SIZE

# Training parameters
epochs = 10
training_batches = 0
validation_batches = 0
batch_size = 32
ewc_lambda = 10.0
ewc_method = EWC_Method.FISHER_MATRIX

# Labels to classify in each task
task_labels = [
    [0,1],
    [2,3],
]

# base model for sequential tasks
# each model gets these layers as a base, then adds own head layers
# i.e. these weights are *shared*
model_inputs = model_layer = tf.keras.Input(shape=model_input_shape)
model_layer = tf.keras.layers.Conv2D(20, (3,3), activation="relu", name="conv2d_0")(model_layer)
model_layer = tf.keras.layers.Conv2D(20, (3,3), activation="relu", name="conv2d_1")(model_layer)
model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.Conv2D(20, (3,3), activation="relu", name="conv2d_2")(model_layer)
model_layer = tf.keras.layers.Conv2D(20, (3,3), activation="relu", name="conv2d_3")(model_layer)
model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.Conv2D(20, (3,3), activation="relu", name="conv2d_4")(model_layer)
model_layer = tf.keras.layers.Conv2D(20, (3,3), activation="relu", name="conv2d_5")(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.GlobalMaxPool2D()(model_layer)
# model_layer = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="conv2d_6")(model_layer)
model_layer = tf.keras.layers.Flatten()(model_layer)
model_layer = tf.keras.layers.Dense(64, activation="relu")(model_layer)
model_layer = tf.keras.layers.Dense(64, activation="relu")(model_layer)
base_model = tf.keras.Model(inputs=model_inputs, outputs=model_layer, name="model")

# Layers specific to each task
# Not shared
task_head_layers = [
    [tf.keras.layers.Dense(len(labels))] for labels in task_labels
]

# The base loss function for tasks
# Currently all tasks have the same structure so only one loss
# Could use a list in future
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

print(f"BASE MODEL SUMMARY")
base_model.summary()

training_image_augmentation = None
training_image_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomZoom(
            height_factor=(-0.05, -0.25),
            width_factor=(-0.05, -0.25)),
    tf.keras.layers.RandomRotation(0.15)
])

# -----------------------------------------------------------------------------
# AUTOMATED SETUP: DON'T TOUCH BELOW HERE UNLESS CONFIDENT
# -----------------------------------------------------------------------------

# Create, compile, and build all models
models = []
for task_index in range(len(task_labels)):
    if task_head_layers is None:
        layers = []
    else:
        layers = task_head_layers[task_index]

    curr_model_layer = model_layer
    for layer in layers:
        curr_model_layer = layer(curr_model_layer)

    curr_model = tf.keras.Model(
        inputs=model_inputs, outputs=curr_model_layer, name=f"task_{task_index+1}_model")
    models.append(curr_model)

# Create the task representations (see SequentialTask)
tasks = []
for task_index in range(len(task_labels)):
    tasks.append(Task(
        name=f"Task {task_index+1}",
        model=models[task_index],
        model_base_loss=loss_fn,
        task_labels=task_labels[task_index],
        training_batches = training_batches,
        validation_batches = validation_batches,
        batch_size=batch_size,
        training_image_augmentation = training_image_augmentation,
        run_eagerly = RUN_EAGERLY
    ))


# Create the manager
manager = SequentialLearningManager(base_model, tasks, epochs, ewc_method, ewc_lambda)
# Train all tasks sequentially
manager.train_all()
base_model.save("models/main_model")
# Plot output data
manager.plot_validation_callback_data(
    "loss", title="Task Total Loss Over All Epochs", ylabel="Total Loss (CategoricalCrossentropy)")
manager.plot_validation_callback_data(
    "base_loss", title="Task Base Losses Over All Epochs", ylabel="Base Loss (CategoricalCrossentropy)")
multiplot_data(manager.get_training_histories())
