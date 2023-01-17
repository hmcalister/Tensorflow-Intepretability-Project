#fmt: off
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from Utilities.Interpretability.InterpretabilityMethods import *
from Utilities.Tasks.IntelNaturalScenesClassificationTask import IntelNaturalScenesClassificationTask as Task
# fmt: on

task_labels = [0,1,2,3,4,5]
image_shape=(128,128)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# base model for sequential tasks
# each model gets these layers as a base, then adds own head layers
# i.e. these weights are *shared*
model_inputs = model_layer = tf.keras.Input(shape=(*image_shape, 3))
model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_0")(model_layer)
model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_1")(model_layer)
model_layer = tf.keras.layers.Conv2D(64, (3,3), activation="relu", name="conv2d_2")(model_layer)
model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.Conv2D(128, (3,3), activation="relu", name="conv2d_3")(model_layer)
model_layer = tf.keras.layers.Conv2D(128, (3,3), activation="relu", name="conv2d_4")(model_layer)
model_layer = tf.keras.layers.Conv2D(128, (3,3), activation="relu", name="conv2d_5")(model_layer)
model_layer = tf.keras.layers.MaxPool2D((2,2))(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.Conv2D(256, (3,3), activation="relu", name="conv2d_6")(model_layer)
model_layer = tf.keras.layers.Conv2D(256, (3,3), activation="relu", name="conv2d_7")(model_layer)
model_layer = tf.keras.layers.Conv2D(256, (3,3), activation="relu", name="conv2d_8")(model_layer)
model_layer = tf.keras.layers.BatchNormalization()(model_layer)
model_layer = tf.keras.layers.Flatten()(model_layer)
model_layer = tf.keras.layers.Dense(64, activation="relu")(model_layer)
model_layer = tf.keras.layers.Dense(64, activation="relu")(model_layer)
model_layer = tf.keras.layers.Dense(len(task_labels))(model_layer)
base_model = tf.keras.Model(inputs=model_inputs, outputs=model_layer, name="model")
model = tf.keras.Model(base_model.input, model_layer)
model.summary()

task = Task(
    name="TestTask",
    data_path="../datasets/IntelNaturalScenes/",
    model=model,
    model_base_loss=loss_fn,
    task_labels=task_labels,
    training_batches=0,
    validation_batches=0,
    batch_size=16,
    image_size=image_shape
)

d = task.validation_dataset.take(1)
images, labels = d.as_numpy_iterator().next()
images = images[:16]
labels = labels[:16]
plot_images(images)

MODEL_SAVE_PATH = "models/vanilla_intel"
training_loop = 0
history_save_path = "vanilla_history.csv"
all_history_df = pd.DataFrame()

while True:
    print("-=*="*40+"-")
    print(f"{training_loop=}")
    print("-=*="*40+"-")
    training_loop += 1

    task.compile_model(loss_fn=loss_fn,
        optimizer = tf.keras.optimizers.Adam(1e-3)
    )
    history = task.train_on_task(epochs=50)
    history_df = pd.DataFrame(history.history)
    all_history_df = pd.concat([all_history_df, history_df], ignore_index=True)
    all_history_df.to_csv(history_save_path)
    model.save(MODEL_SAVE_PATH)