# fmt: off
import pandas as pd
from Utilities.Utils import *
from Utilities.Tasks.CIFAR10ClassificationTask import CIFAR10ClassificationTask as Task
from Utilities.SequentialLearning.AdversarialTraining import AdversarialExampleTrainer
from Utilities.SequentialLearning.EWC_Methods.EWC_Methods import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

MODEL_SAVE_PATH = "models/CIFAR10_INTERLEAVED_ADVERSARIAL_MODEL"
HISTORY_SAVE_PATH = "history.csv"
# True for easier debugging
# False for compiled models, faster train time
RUN_EAGERLY: bool = False

print(f"GPU: {tf.config.list_physical_devices('GPU')}")
model_input_shape = Task.IMAGE_SIZE

# Training parameters
epochs = 100
training_batches = 0
validation_batches = 0
batch_size = 32

# Labels to classify in each task
task_labels = [
    i for i in range(10)
]

# base model for sequential tasks
# each model gets these layers as a base, then adds own head layers
# i.e. these weights are *shared*
model_inputs = model_layer = tf.keras.Input(shape=model_input_shape)
model_layer = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="conv2d_0")(model_layer)
model_layer = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="conv2d_1")(model_layer)
model_layer = tf.keras.layers.Conv2D(32, (3,3), activation="relu", name="conv2d_2")(model_layer)
model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_3")(model_layer)
model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_4")(model_layer)
# model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_5")(model_layer)
# model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
# model_layer = tf.keras.layers.BatchNormalization()(model_layer)
# model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_6")(model_layer)
# model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_7")(model_layer)
# model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_8")(model_layer)
# model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
# model_layer = tf.keras.layers.BatchNormalization()(model_layer)
# model_layer = tf.keras.layers.Conv2D(128, (3,3), activation="relu", name="conv2d_9")(model_layer)
# model_layer = tf.keras.layers.Conv2D(128, (3,3), activation="relu", name="conv2d_10")(model_layer)
# model_layer = tf.keras.layers.Conv2D(128, (3,3), activation="relu", name="conv2d_11")(model_layer)
model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.Flatten()(model_layer)
model_layer = tf.keras.layers.Dense(64, activation="relu")(model_layer) 
model_layer = tf.keras.layers.Dense(64, activation="relu")(model_layer)
model_layer = tf.keras.layers.Dense(len(task_labels))(model_layer)
base_model = tf.keras.Model(inputs=model_inputs, outputs=model_layer, name="model")

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

task = Task(
        name=f"Task 1",
        model=base_model,
        model_base_loss=loss_fn,
        task_labels=task_labels,
        training_batches = training_batches,
        validation_batches = validation_batches,
        batch_size=batch_size,
        training_image_augmentation = training_image_augmentation,
        run_eagerly = RUN_EAGERLY,
        # data_path="../../datasets/IntelNaturalScenes"
    )

EPSILON = 0.01
adversarial_example_trainer = AdversarialExampleTrainer(task)

all_history = pd.DataFrame()
try:
    for iteration_index in range(1,100):
        print(f"{'-='*80}")
        print(f"{iteration_index=}")
        print(f"{'-='*80}")

        history = task.train_on_task(epochs=10)
        history = pd.DataFrame(history.history)
        history["TrainType"] = "Vanilla"
        all_history = pd.concat([all_history, history], ignore_index=True)
        all_history.to_csv(HISTORY_SAVE_PATH)
        base_model.save(MODEL_SAVE_PATH)        

        history = adversarial_example_trainer.train_adversarial(epsilon=EPSILON, epochs=1)
        history = pd.DataFrame(history.history)
        history["TrainType"] = "Adversarial"
        all_history = pd.concat([all_history, history], ignore_index=True)
        all_history.to_csv(HISTORY_SAVE_PATH)
        base_model.save(MODEL_SAVE_PATH)        
except KeyboardInterrupt:
    print("KEYBOARD INTERRUPT")

base_model.save(MODEL_SAVE_PATH)
plt.plot(all_history["loss"], label="loss")
plt.plot(all_history["val_loss"], label="val_loss")
plt.title("Interleaved Adversarial Training")
plt.ylabel("Categorical Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
