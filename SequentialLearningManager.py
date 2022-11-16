# fmt: off
from typing import List, Union
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
            print(f"FINISHED VALIDATION\n")

    def __init__(self, base_model: tf.keras.models.Sequential,
                 tasks: List[SequentialTask],
                 epochs: Union[int, List[int]],
                 EWC_method=None):
        """
        Create a new SequentialLearningManager

        Parameters:
            base_model: tf.keras.models.Sequential
                The base model (or more accurately, a model with all shared weights)
            tasks: List(SequentialTask):
                The list of tasks to learn on (ordered in a list).
                See SequentialTask class for more details
            EWC_method:
                Method to calculate elastic weight consolidation importances
                Presence also adds EWC term to subsequent loss functions
        """

        self.base_model:tf.keras.models.Sequential = base_model
        self.tasks: List[SequentialTask] = tasks
        self.EWC_terms: List[EWCTerm] = []
        self.training_histories: List[tf.keras.callbacks.History] = []
        self._current_task_index: int = 0

        self.validation_callback = \
            SequentialLearningManager.SequentialValidationCallback(tasks)

        self.epochs = epochs
        if isinstance(epochs, int):
            self.epochs: List[int] = [epochs for _ in range(len(self.tasks))]

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

        # Create Loss function for next task
        base_loss_function = current_task.model_base_loss
        def EWC_loss(y_true, y_pred):
            loss = base_loss_function(y_true, y_pred)
            for ewc_term in self.EWC_terms:
                loss += ewc_term.create_loss(self.base_model)
            return loss

        # Recompile model to use new loss
        # Note we keep the metrics!
        current_task.compile_model(EWC_loss)

        # Train model, store history
        history = current_task.train_on_task(epochs=self.epochs[self._current_task_index],
                                          callbacks=[self.validation_callback])

        self.training_histories.append(history)
        self._current_task_index += 1

    def plot_task_training_histories(self):
        multiplotData(self.training_histories)

    def plot_validation_callback_data(self, key:str, title=None, ylabel=None):
        """
        Plot the data from the validation callback
        Possible keys are any metric name or 'loss', e.g.
        key is in the set {"loss", "base_loss"} 
        """

        plt.plot(self.validation_callback.validation_results[key], marker='o')
        plt.title(title)
        plt.xlabel("Epochs")
        plt.ylabel(ylabel)
        plt.legend([t.name for t in self.tasks])
        task_boundaries = np.cumsum(self.epochs)
        plt.vlines(task_boundaries, colors="k",
            ymin=np.min(self.validation_callback.validation_results[key]),
            ymax=np.max(self.validation_callback.validation_results[key]),
            linestyles="dashed", alpha=0.5)
        plt.show()