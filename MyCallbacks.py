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

    def __init__(self, model: tf.keras.models.Sequential,
                 x_lim: Tuple[float, float],
                 x_step: float,
                 y_lim: Tuple[float, float], 
                 input_data_fn,
                 data_fn=None):
        """
        Create a new instance of this call back.

        Parameters:
            model: tf.keras.models.Sequential
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
        self.input_tensors = np.apply_along_axis(input_data_fn,0,self.var_input).T
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
        modelLine = plt.plot([],[], label="Model Output")[0]
        trueDataLine = plt.plot([],[], label="True Data")[0]
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        def animInit():
            modelLine.set_data([],[])
            trueDataLine.set_data(self.var_input, self.trueData)
            return [modelLine, trueDataLine]

        def animate(i):
            legend.get_texts()[0].set_text(f"Time: {i}")
            modelLine.set_data(self.var_input, self.model_outputs[i])
            trueDataLine.set_data(self.var_input, self.trueData)
            return [modelLine, trueDataLine, legend]

        anim = animation.FuncAnimation(fig, animate, init_func=animInit,
            frames=range(0, len(self.model_outputs), 1+skip_factor),
            interval=interval, blit=True, repeat_delay=1000)

        plt.tight_layout()
        plt.show()
        
        if save or input("Save Animation? (Y/N) ").upper().startswith("Y"):
            try:
                file_name = input("FILENAME (default: animation) ")
                print("SAVING: PRESS ^C TO ABORT")
                anim.save("images/"+file_name+".gif", fps=1000/interval)
            except KeyboardInterrupt:
                print("ABORT ANIMATION SAVE")


class WeightTrackingCallback(tf.keras.callbacks.Callback):
    """
    A callback to track weights during training.
    Weights can be tracked by keeping a direct history, checking sign flipping, etc...
    Each tracking method is implemented in a child class
    """

    def __init__(self, model: tf.keras.models.Sequential):
        """
        Create a new instance of this call back.

        Parameters:
        model: tf.keras.models.Sequential
                        The model to track weights of.
                        Must be passed so lifecycle hooks can reference model!
        """

        super().__init__()
        self.model = model


class SignFlippingTracker(WeightTrackingCallback):
    """
    Track the weights of a network by checking the sign of those weights after each epoch
    Count how often each weight flips sign
    """

    class MeasurePeriod(Enum):
        BATCH_END = 1,
        EPOCH_END = 2

    def __init__(self, model: tf.keras.models.Sequential,
                 measurePeriod: MeasurePeriod = MeasurePeriod.BATCH_END):
        """
        Create a new instance of this call back.

        Parameters:
        model: tf.keras.models.Sequential
                        The model to track weights of.
                        Must be passed so lifecycle hooks can reference model!
        """

        super().__init__(model)
        self.measurePeriod = measurePeriod
        self.signChanges = []
        self.currWeights = []
        for layer in self.model.layers:
            weights = layer.get_weights()
            self.currWeights.append((weights[0], weights[1]))
            w = np.zeros_like(weights[0])
            b = np.zeros_like(weights[1])
            self.signChanges.append([w, b])

    def measureWeightChanges(self):
        updatedWeights = []
        for i, layer in enumerate(self.model.layers):
            currLayer = self.currWeights[i]
            weights = layer.get_weights()
            updatedWeights.append((weights[0], weights[1]))

            # Multiply new weights with old weights
            # If a sign changes has occured, this is represented as a -1 after np.sign
            # So check for equality with -1 to find flipped signs
            w_changes = np.sign(weights[0] * currLayer[0]) == -1
            b_changes = np.sign(weights[1] * currLayer[1]) == -1

            self.signChanges[i][0] += w_changes
            self.signChanges[i][1] += b_changes
        self.currWeights = updatedWeights

    def on_batch_end(self, epoch, logs=None):
        if self.measurePeriod == self.MeasurePeriod.BATCH_END:
            self.measureWeightChanges()

    def on_epoch_end(self, epoch, logs=None):
        if self.measurePeriod == self.MeasurePeriod.EPOCH_END:
            self.measureWeightChanges()
