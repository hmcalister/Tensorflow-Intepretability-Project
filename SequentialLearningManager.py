# fmt: off
from typing import List, Union
import typing
import numpy as np
import matplotlib.pyplot as plt
from SequentialTask import SequentialTask
from EWCMethods import *
from MyUtils import *
from MyCallbacks import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on



class SequentialLearningManager():
    """
    A manager for sequential learning environments. Takes a model
    and a list of tasks to train on (ordered). Tasks must be an object
    (SequentialTask) that offers a task head (placed at end of model) the
    task data (tf.keras.Dataset) and some validation data to be tested each epoch.
    """

    class SequentialValidationCallback(tf.keras.callbacks.Callback):
        def __init__(self, tasks: List[SequentialTask]):
            """
            Create a new validation callback, checking model performance 
            on all task each epoch

            Parameters:
                tasks: List(SequentialTask):
                    The list of tasks to learn on (ordered in a list).
                    See SequentialTask class for more details
            """
            super().__init__()
            self.tasks = tasks
            self.validation_results = {}

        def on_epoch_end(self, epoch, logs=None):
            epoch_results = {}
            print(f"\nVALIDATING AFTER EPOCH {epoch+1}")
            for task_index in range(len(self.tasks)):
                task_results: dict = self.tasks[task_index].evaluate_model()
                for k,v in task_results.items():
                    if k not in epoch_results:
                        epoch_results[k] = [v]
                    else:
                        epoch_results[k].append(v)
            for k,v in epoch_results.items():
                if k in self.validation_results:
                    self.validation_results[k].append(v)
                else:
                    self.validation_results[k] = [v]
            print(f"FINISHED VALIDATION")

    def __init__(self, base_model: tf.keras.models.Model,
                 tasks: List[SequentialTask],
                 epochs: Union[int, List[int]],
                 EWC_method: EWC_Method = EWC_Method.NONE,
                 log_file_path="logs/Manager.log"):
        """
        Create a new SequentialLearningManager

        Parameters:
            base_model: tf.keras.models.Model
                The base model (or more accurately, a model with all shared weights)
            tasks: List(SequentialTask):
                The list of tasks to learn on (ordered in a list).
                See SequentialTask class for more details
            EWC_method:
                Method to calculate elastic weight consolidation importance's
                Presence also adds EWC term to subsequent loss functions
            log_file_path: String
                The path to the log file to be used
                Defaults to logs/Manager.log
        """

        self.base_model:tf.keras.models.Model = base_model
        self.base_model_layers: List[tf.keras.layers.Layer] = [l for l in base_model.layers]
        self.tasks: List[SequentialTask] = tasks
        self.EWC_terms: List[EWC_Term] = []
        self.training_histories: List[tf.keras.callbacks.History] = []
        self._current_task_index: int = 0
        self.EWC_term_creator = EWC_Term_Creator(EWC_method, base_model, tasks)

        self.validation_callback = \
            SequentialLearningManager.SequentialValidationCallback(tasks)

        # We're doing a little hack here so type hinting is thrown off
        # After this block, epochs is type List[int]
        self.epochs = epochs  # type: ignore
        if isinstance(epochs, int):
            self.epochs: List[int] = [epochs for _ in range(len(self.tasks))]

        self._log_file:typing.TextIO = open(log_file_path, "wt")

    def train_all(self):
        """
        Train all tasks, sequentially
        """

        while self._current_task_index < len(self.tasks):
            print(f"---***--- {self.tasks[self._current_task_index].name} ---***---")
            self.train_next_task()
        
    def train_next_task(self):
        """
        Begin training on the next task in the list, or return None if no such task exists
        """

        if self._current_task_index >= len(self.tasks):
            return None

        current_task = self.tasks[self._current_task_index]

        # Quick log of all EWC weights 
        self.log_to_file(f"{'='*80}")
        self.log_to_file(f"TASK {self._current_task_index}")
        for term_index, ewc_term in enumerate(self.EWC_terms):
                self.log_to_file(f"{'-'*80}")
                self.log_to_file(f"TERM {term_index}")
                for layer_index, layer in enumerate(ewc_term.optimal_weights):
                    self.log_to_file(f"LAYER {layer_index}")
                    for tensor in layer:
                        self.log_to_file(f"{tensor}")
        self.log_to_file(f"{'='*80}")

        # Recompile model to use new loss
        # Note we keep the metrics!
        for task in self.tasks:
            base_loss_function = task.model_base_loss
            task.compile_model(EWC_Loss(base_loss_function, self.base_model_layers, self.EWC_terms))

        # Train model, store history
        history = current_task.train_on_task(epochs=self.epochs[self._current_task_index],
                                          callbacks=[self.validation_callback, *self.EWC_term_creator.callback_dict.values()])
        self.training_histories.append(history)

        # Create an EWC term for the now completed task
        self.EWC_terms.append(self.EWC_term_creator.create_term(lam=1))
        self._current_task_index += 1

    def plot_task_training_histories(self):
        multiplot_data(self.training_histories)

    def plot_validation_callback_data(self, key:str, title: str="", ylabel: str=""):
        """
        Plot the data from the validation callback
        Possible keys are any metric name or 'loss', e.g.
        key is in the set {"loss", "base_loss"} 
        """

        data = self.validation_callback.validation_results[key]
        marker=None
        markersize=5
        if len(data) < 100:
            marker = 'o'

        fig = plt.figure(figsize=(12,8))
        plt.plot(data, marker=marker, markersize=markersize)
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend([t.name for t in self.tasks])
        task_boundaries = np.cumsum(self.epochs)-1
        plt.vlines(task_boundaries, colors="k",  # type: ignore
            ymin=np.min(data),
            ymax=np.max(data),  # type: ignore
            linestyles="dashed", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def log_to_file(self, log_string: typing.Any):
        """
        A quick and overly simple logging function to dump something to a file
        Separates something potentially interesting from screens of TF logs

        Parameters:
            log_string: str
                The string to append to the log file
        """

        self._log_file.write(log_string)
        self._log_file.write("\n")