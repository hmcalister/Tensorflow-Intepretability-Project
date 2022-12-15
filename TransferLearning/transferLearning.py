#fmt: off
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from Utilities.Interpretability.InterpretabilityMethods import *
from Utilities.SequentialLearning.Tasks.Stl10ClassificationTask import Stl10ClassificationTask as Task
# fmt: on

task_labels = [0,1,2]

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

base_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=Task.IMAGE_SIZE
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
    model=model,
    model_base_loss=loss_fn,
    task_labels=task_labels,
    training_batches=0,
    validation_batches=0,
    batch_size=32
)

d = task.validation_dataset.take(1)
images, labels = d.as_numpy_iterator().next()
images = images[:16]
labels = labels[:16]
plot_images(images)

KERNEL_INSPECTION_PATH = "images/kernel_inspection/vgg16_vanilla"
task.train_on_task(epochs=25)
MODEL_SAVE_PATH = "models/VGG16_vanilla"
model.save(MODEL_SAVE_PATH)

KERNEL_INSPECTION_PATH = "images/kernel_inspection/vgg16_finetuned"
base_model.trainable = True
base_model.summary()
task.compile_model(loss_fn=loss_fn,
    optimizer = tf.keras.optimizers.Adam(1e-5)
)
task.train_on_task(epochs=25)
MODEL_SAVE_PATH = "models/VGG16_finetuned"
model.save(MODEL_SAVE_PATH)