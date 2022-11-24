# fmt: off
from enum import Enum
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on


class ModelOutputAnimationCallback(tf.keras.callbacks.Callback):
    """
    A callback to measure the model output after each epoch to animate
    """

    def __init__(self, model: tf.keras.models.Model,
                 x_lim: Tuple[float, float],
                 x_step: float,
                 y_lim: Tuple[float, float],
                 input_data_fn,
                 data_fn=None):
        """
        Create a new instance of this call back.

        Parameters:
            model: tf.keras.models.Model
                The model to track weights of.
                Must be passed so lifecycle hooks can reference model!

            x_lim: Tuple(float, float)
                The limits on the input, e.g. (-1,1).

            x_step: float
                The step of the uniform range across input_lim.

            y_lim: Tuple(float, float)
                The y limits of the animation

            input_data_gen: fn
                The function to map a single input to an entire input vector

            data_fn: fn
                The function that generated the training data. If not none then
                plot the "real" function over each domain
        """

        super().__init__()
        self.model = model
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.var_input = np.array([np.arange(x_lim[0], x_lim[1], x_step)])
        self.input_tensors = np.apply_along_axis(
            input_data_fn, 0, self.var_input).T
        self.model_outputs = []
        self.trueData = []
        if data_fn != None:
            self.trueData = data_fn(self.var_input)

    def on_batch_end(self, epoch, logs=None):
        self.model_outputs.append(self.model(self.input_tensors))

    def plot_animation(self, skip_factor=0, interval=20, save=False):
        """
        Plot the actual animation

        Params:
            skip_factor: The number of frames to skip between each frame in the animation

            interval: Duration of each frame
        """

        fig = plt.figure()
        SCALE_FACTOR = 1.1
        plt.axes(xlim=(self.x_lim[0]*SCALE_FACTOR, self.x_lim[1]*SCALE_FACTOR),
                 ylim=(self.y_lim[0]*SCALE_FACTOR, self.y_lim[1]*SCALE_FACTOR))

        legend_trick = plt.plot([], [], color="gold", label=f" ")
        modelLine = plt.plot([], [], label="Model Output")[0]
        trueDataLine = plt.plot([], [], label="True Data")[0]
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        def animInit():
            modelLine.set_data([], [])
            trueDataLine.set_data(self.var_input, self.trueData)
            return [modelLine, trueDataLine]

        def animate(i):
            legend.get_texts()[0].set_text(f"Time: {i}")
            modelLine.set_data(self.var_input, self.model_outputs[i])
            trueDataLine.set_data(self.var_input, self.trueData)
            return [modelLine, trueDataLine, legend]

        anim = animation.FuncAnimation(fig, animate, init_func=animInit,
                                       frames=range(
                                           0, len(self.model_outputs), 1+skip_factor),
                                       interval=interval, blit=True, repeat_delay=1000)

        plt.tight_layout()
        plt.show()

        if save or input("Save Animation? (Y/N) ").upper().startswith("Y"):
            try:
                file_name = input("FILENAME (default: animation) ")
                print("SAVING: PRESS ^C TO ABORT")
                anim.save("images/"+file_name+".gif", fps=int(1000/interval))
            except KeyboardInterrupt:
                print("ABORT ANIMATION SAVE")