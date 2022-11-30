# Tensorflow Sequential Learning
An implementation of sequential learning in Tensorflow with Keras Sequential models, including infrastructure for elastic weight consolidation techniques.

Sequential learning involves having a single model (or multiple models with some shared sections) learn a series of tasks in sequence i.e. learn task A without knowledge of task B, then learn task B without access to the training data of task A. The goal is to prevent models "forgetting" (perhaps, catastrophically) previous tasks when new tasks are introduced

## Project Architecture
Models are created and structured to have a base model (shared weights) with additional task-specific layers placed on top. Models are first created (with the tensorflow functional API to share layers) then passed to SequentialTasks (see `SequentialLearning/SequentialTasks`, and below) which manage an individual task, abstracting away the sequential learning aspect and focus on a single model on a single dataset. The SequentialLearningManager handles sequencing tasks one after the other as well as orchestrating testing of tasks and any sequential learning techniques such as EWC.

### Sequential Task
Conceptually holds all information for a single task in a learning environment. Holds a reference to the task model (the base model plus this tasks extra layers), the training and validation datasets for this task, and the loss function. Holds responsibility for compiling the model (see EWC section below) as well as explicitly calling the `fit` method to train the model. Has methods for evaluating the model on validation data which is set up to be called at the end of each epoch.

Also has several seemingly superfluous attributes (x_lim, y_lim, data_fn...). These attributes are present for the possibility of adding functionality for graphing model input/outputs for interpretability. Currently these are unused and may be removed in the future.

### SequentialLearningManager
Manages learning tasks sequentially. Holds reference to all tasks, as well as being the main interface for the main script. Instantiated with a list of SequentialTasks (with models already present) and parameters for behavior of elastic weight consolidation.

Includes a callback for managing task validation each epoch, so task performance can be graphed over all epochs for each task. On a technical note, this callback data is taken directly from `model.evaluate(..., return_dict=True)`, so the stored data is a dictionary with keys being the loss+metric names and values being tuples of task performances each epoch.

### EWC_Methods
Holds all implementations of elastic weight consolidation in this project. Callbacks for data collection (relating to EWC), methods for calculating the weight importance matrix Omega, and anything else is present here. Each elastic weight consolidation method has a unique way of calculating (or approximating, or guessing) weight importance. The `EWC_Methods.py` file acts as a simple interface to these methods, allowing different methods to be used in a plug-and-play manner by selecting a method from the `EWC_Method` Enum within.

## Interpretability
A second focus of this project is on interpretability of neural networks (particularly convolutional networks, but other network architectures may be investigated in future). This module provides some methods to probe a network for interpretable features, including kernel activation, occlusion sensitivity, and GRADCAM. 

This module also includes methods in the `ModelAugmentation.py` file that allows for altering tensorflow models based on the weight importance calculated during elastic weight consolidation. The goal here is to see if weight importance can provide a useful insight into model interpretability, or boost current interpretability measures.

