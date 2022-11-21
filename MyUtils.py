# fmt: off
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

# Get the set of all powers of x like so [x**1, x**2, x**3,...]
def powers_set(max_power: int):
    return lambda x: np.array([x[0]**i for i in range(1, max_power+1)])

def plot_history(history, keys: dict, title:str="", xlabel:str="", ylabel:str="", yscale:str="linear"):
    """
    Plot the history of a model's training.

    Parameters:
    history: The history object returned from model.fit
    keys: A dictionary from string to string. 
        The key is the attribute in history to plot
        The value is the label for the legend
    title: Title of the plot. Defaults to None.
    xlabel: The xlabel of the plot. Defaults to None.
    ylabel: The ylabel of the plot. Defaults to None.
    yscale: The yscale of the plot. Defaults to Linear.
    """
    
    legend = []
    for (k, v) in keys.items():
        plt.plot(history.history[k])
        legend.append(v)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.yscale(yscale)  # type: ignore
    plt.legend(legend, loc='upper right')
    plt.tight_layout()
    plt.show()


def plotFunctionSingleVariableInputs(model, 
    input_shape, input_lim, step, 
    static_input_val = 0, data_fn=None):
    """
    Plot the output (single dimension) of a model with only one variable input

    Parameters
    model: The tensorflow model to plot outputs of
    input_shape: The shape of the input to the model
    input_lim: The limits on the input, e.g. (-1,1)
    step: The step of the uniform range across input_lim
    static_input_val: The value to set other inputs to. Defaults to 0
    data_fn: The function that generated the training data. If not none then
        plot the "real" function over each domain
    """

    var_input = np.arange(input_lim[0], input_lim[1], step)
    input_tensor_shape = (len(var_input), *input_shape)
    static_tensor = np.zeros(input_tensor_shape)
    static_tensor.fill(static_input_val)
    true_data = []
    outputs = []

    for input_index in range(input_shape[0]):
        current_input_tensor = static_tensor.copy()
        current_input_tensor[:,input_index] = var_input

        result = model(current_input_tensor)
        outputs.append(result)

        if data_fn is not None:
            result = np.apply_along_axis(data_fn, 1, current_input_tensor)
            true_data.append(result)

    fig = plt.figure(1)
    total_cols = int(len(outputs)**0.5)
    total_rows = len(outputs) // total_cols
    if len(outputs) % total_cols != 0: total_rows += 1

    for i in range(1,len(outputs)+1):
        ax = fig.add_subplot(total_rows, total_cols, i)
        ax.plot(var_input, outputs[i-1])
        if data_fn is not None:
            ax.plot(var_input, true_data[i-1], "--")
        ax.set_xlabel(f"Input {i}")

    fig.tight_layout()
    plt.show()

def plotFunctionCombinedInputs(model, 
    input_lim, step, 
    input_data_fn, data_fn=None):
    """
    Plot the output (single dimension) of a model with only one variable input

    Parameters
    model: The tensorflow model to plot outputs of
    input_lim: The limits on the input, e.g. (-1,1)
    step: The step of the uniform range across input_lim
    input_data_gen: The function to map a single input to an entire input vector
    data_fn: The function that generated the training data. If not none then
        plot the "real" function over each domain
    """

    var_input = np.arange(input_lim[0], input_lim[1], step)
    input_tensors = input_data_fn(var_input).T
    model_outputs = model(input_tensors)

    plt.plot(var_input, model_outputs, label="Model Output")

    if data_fn != None:
        true_data = data_fn(var_input.T)
        plt.plot(var_input, true_data, label="True Data")
        plt.legend()
    plt.show()

def multiplot_data(data, xlabels: List = []):
    fig = plt.figure(1, figsize=(16,8))
    total_cols = int(len(data)**0.5)
    total_rows = len(data) // total_cols
    if len(data) % total_cols != 0: total_rows += 1


    for i in range(1,len(data)+1):
        ax = fig.add_subplot(total_rows, total_cols, i)

        curr_data = data[i-1]

        if isinstance(curr_data, tf.keras.callbacks.History):
            ax.set_title(f"Task {i}")
            legend = []
            for k,v in curr_data.history.items():
                ax.plot(v, label=k)
                
            ax.legend()

        if isinstance(curr_data, dict):
            for k,v in curr_data.items():
                ax.plot(v)
        
        if isinstance(curr_data, list):
            ax.plot(curr_data)
        
        if len(xlabels) > i-1:
            ax.set_xlabel(xlabels[i-1])

    fig.tight_layout()
    plt.show()

