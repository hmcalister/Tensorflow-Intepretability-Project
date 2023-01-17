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


base_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(*image_shape,3)
)
base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(64, "relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(64, "relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(len(task_labels))(x)
model = tf.keras.Model(base_model.input, x)
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

MODEL_SAVE_PATH = "models/transfer_intel"
training_loop = 0
history_save_path = "history.csv"
all_history_df = pd.DataFrame()

while True:
    print("-=*="*40+"-")
    print(f"{training_loop=}")
    print("-=*="*40+"-")
    training_loop += 1

    # Train for some epochs with only head layers
    base_model.trainable = False
    task.compile_model(loss_fn=loss_fn,
        optimizer = tf.keras.optimizers.Adam(1e-3)
    )
    history = task.train_on_task(epochs=40)
    history_df = pd.DataFrame(history.history)
    all_history_df = pd.concat([all_history_df, history_df], ignore_index=True)
    all_history_df.to_csv(history_save_path)
    model.save(MODEL_SAVE_PATH)

    # Train for some epochs with fine tuning
    base_model.trainable = True
    task.compile_model(loss_fn=loss_fn,
        optimizer = tf.keras.optimizers.Adam(1e-5)
    )
    history = task.train_on_task(epochs=10)
    history_df = pd.DataFrame(history.history)
    all_history_df = pd.concat([all_history_df, history_df], ignore_index=True)
    all_history_df.to_csv(history_save_path)
    model.save(MODEL_SAVE_PATH)
