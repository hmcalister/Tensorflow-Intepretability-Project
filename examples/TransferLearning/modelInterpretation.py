#fmt: off
import os
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from Utilities.Interpretability.InterpretabilityMethods import *
from Utilities.Tasks.IntelNaturalScenesClassificationTask import IntelNaturalScenesClassificationTask as Task
# fmt: on

MODEL_SAVE_PATH = "models/vanilla_intel"

model:tf.keras.models.Model = tf.keras.models.load_model(MODEL_SAVE_PATH, compile=False) #type:ignore
model.summary()

KERNEL_INSPECTION_PATH = "images/kernel_inspection/vanilla_intel"
Path(KERNEL_INSPECTION_PATH).mkdir(exist_ok=True, parents=True)
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        print(layer.name)
        kernel_activations(model, layer_name=layer.name, steps=500, step_size=0.01, filters_per_plot=16, save_path=KERNEL_INSPECTION_PATH)