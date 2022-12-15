#fmt: off
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from Utilities.Interpretability.InterpretabilityMethods import *
from Utilities.SequentialLearning.Tasks.Stl10ClassificationTask import Stl10ClassificationTask as Task
# fmt: on

VANILLA_MODEL_SAVE_PATH = "models/VGG16_vanilla"
FINETUNED_MODEL_SAVE_PATH = "models/VGG16_finetuned"

vanilla_model:tf.keras.models.Model = tf.keras.models.load_model(VANILLA_MODEL_SAVE_PATH, compile=False) #type:ignore
finetuned_model:tf.keras.models.Model = tf.keras.models.load_model(FINETUNED_MODEL_SAVE_PATH, compile=False) #type:ignore

KERNEL_INSPECTION_PATH = "images/kernel_inspection/vgg16_vanilla"
for layer in vanilla_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        print(layer.name)
        kernel_activations(vanilla_model, layer_name=layer.name, steps=500, step_size=0.01, filters_per_plot=16, save_path=KERNEL_INSPECTION_PATH)
        # kernel_inspection(vanilla_model, layer_name=layer.name, filters_per_plot=16, save_path=KERNEL_INSPECTION_PATH)

KERNEL_INSPECTION_PATH = "images/kernel_inspection/vgg16_finetuned"
for layer in finetuned_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        print(layer.name)
        kernel_activations(finetuned_model, layer_name=layer.name, steps=500, step_size=0.01, filters_per_plot=16, save_path=KERNEL_INSPECTION_PATH)
        # kernel_inspection(finetuned_model, layer_name=layer.name, filters_per_plot=16, save_path=KERNEL_INSPECTION_PATH)