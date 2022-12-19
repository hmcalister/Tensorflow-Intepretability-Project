#fmt: off
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from Utilities.Interpretability.InterpretabilityMethods import *
from Utilities.SequentialLearning.Tasks.Stl10ClassificationTask import Stl10ClassificationTask as Task
# fmt: on

MODEL_SAVE_PATH = "models/VGG16_transfer"

model:tf.keras.models.Model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False) #type:ignore

KERNEL_INSPECTION_PATH = "images/kernel_inspection/vgg16_transfer"
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        print(layer.name)
        kernel_activations(model, layer_name=layer.name, steps=500, step_size=0.01, filters_per_plot=16, save_path=KERNEL_INSPECTION_PATH)