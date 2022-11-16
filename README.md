# Tensorflow Sequential Learning
An implementation of sequential learning in Tensorflow with Keras Sequential models, including infrastructure for elastic weight consolidation techniques (in progress).

## Project Architecture
Models are created in the `main.py` file and are structured to have a base model (shared weights) with additional task-specific layers placed on top.

### Sequential Task
Conceptually holds all information for a single task in a learning environment. Holds a reference to the task model (the base model plus this tasks extra layers), the training and validation datasets for this task, and the loss function. Holds responsibility for compiling the model (see EWC section below) as well as explicitly calling the `fit` method to train the model. Has methods for evaluating the model on validation data which is set up to be called at the end of each epoch.

Also has several seemingly superfluous attributes (x_lim, y_lim, data_fn...). These attributes are present for the possibility of adding functionality for graphing model input/outputs for interpretability.

### SequentialLearningManager
Manages learning tasks sequentially. Holds reference to all tasks, as well as being the main interface for the (unfinished) elastic weight consolidation methods.

Includes a callback for managing task validation each epoch, so task performance can be graphed over all epochs for each task. On a technical note, this callback data is taken directly from `model.evaluate(..., return_dict=True)`, so the stored data is a dictionary with keys being the loss+metric names and values being tuples of task performances each epoch.

### EWCMethods
Holds all implementations of elastic weight consolidation in this project. Callbacks for data collection (relating to EWC), methods for calculating the weight importance matrix Omega, and anything else is present here. This section of the project is under development and may change rapidly.

### MyCallbacks and MyUtils
These files include some additional keras callbacks and matplotlib wrapper methods for ease of use. In general these can be understood from documentation in the classes/methods but are not essential for understanding this project. Contents of these files are subject to rapid change.

