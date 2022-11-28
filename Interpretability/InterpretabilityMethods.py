# fmt: off
from typing import List
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

def plot_images(images: List[np.ndarray], titles:List[str]=[], cmap: str = "viridis"):
    """
    Plot a series of images in a grid using matplotlib and subplots
    Useful to show interpretations of different filters in conv-nets

    Parameters:
        images: List[np.ndarray]
            The list of images to be plotted
            Images (items of the array) should be square, etc.. and ar passed directly to imshow
            The list itself can be any length (even non-square)

        titles: List[str]
            Title each image. Defaults to empty array.
            If empty, no titles are added

        cmap: str
            The colour map to use for the images
            Default is viridis 
    """
    total_cols = int(len(images)**0.5)
    total_rows = len(images) // total_cols
    if len(images) % total_cols != 0: total_rows += 1

    fig = plt.figure()
    for i in range(0,len(images)):
        ax = fig.add_subplot(total_rows, total_cols, i+1)
        ax.imshow(images[i], cmap=cmap)
        ax.axis("off")
        if i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()

def kernel_inspection(
        model: tf.keras.Model,
        layer_index: int,
        steps:int = 500,
        step_size = 0.01,
    ):
    """
    Visualize the convolution filters of a layer from a model
    Uses a gradient ascent on random noise to get information on filter operation

    Based on the example from: https://www.sicara.fr/blog-technique/2019-08-28-interpretability-deep-learning-tensorflow
            Subsection: Kernel Inspection

    Parameters:
        model: tf.keras.Model
            The model to sample layers from
        layer_index: int
            An index into the supplied model's layers
            Layer at this index should be a conv2d layer
        steps: int
            The number of steps to iterate over when building a visualization
            More steps gives a smoother/better interpretation
        step_size: float
            The influence of each step on the visualisation
            Smaller steps results in longer wait times but more stable visuals
    """

    target_layer = model.layers[layer_index]
    print(target_layer.name)
    print()
    submodel = tf.keras.models.Model(
        model.input,
        [target_layer.output]
    )
    filter_results = []
    # Get input shape and replace None (batch placeholder) with 1
    noise_shape = submodel.input_shape
    noise_shape = (1, *noise_shape[1:])
    for filter_index in range(target_layer.filters):
        print(f"OPERATING ON FILTER: {filter_index+1}", end="\r")
        # Make random noise the layer can use
        x: tf.Variable = tf.Variable(tf.cast(np.random.random(noise_shape), tf.float32))
        for _ in range(steps):
            with tf.GradientTape() as tape:
                y: tf.Tensor = submodel(x)  # type: ignore
                loss = tf.reduce_mean(y[:,:,:,filter_index])  # type: ignore
            grads = tape.gradient(loss, x)
            normalized_grads = grads/(tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
            x.assign_add( normalized_grads*step_size )
        filter_results.append(x[0,:,:,:])  # type: ignore

    plot_images(filter_results, 
        titles=[f"Filter {i+1}" for i in range(0,len(filter_results))],
        cmap="gray"
    )

def _process_occlude_image(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    label: tf.Tensor,
    patch_size: int,
    stride: int
):
    # Convert the image to something more... manageable
    # Remove first index (added from tf dataset)
    # convert to numpy array
    image: np.ndarray = np.array(image_tensor)  # type: ignore
    sensitivity_map = np.zeros((image.shape[0], image.shape[1]))
    patched_images = []
    patch_value = np.average([np.max(image), np.min(image)])
    # Loop over every possible position of the occulsion square
    # note for now the stride is simply the patch size, i.e.
    # disjoint patches
    for top_left_y in range(0, image.shape[0],stride):
        for top_left_x in range(0, image.shape[1],stride):
            # Copy the original image, apply a square of black over that patch
            patched_image = np.array(image, copy=True)
            patched_image[
                top_left_y:top_left_y+patch_size,
                top_left_x:top_left_x+patch_size, 
                : 
            ] = patch_value
            # We collect all the patched images together to be processed in a single batch
            patched_images.append(patched_image)
    patched_images = np.array(patched_images)
    # Note multiplying by label, we are only interested in 
    # model predicted likelihood of correct class
    predictions = model(patched_images) * label # type: ignore
    # Loop over predictions and apply confidence to the sensitivity map
    prediction_index = 0
    for top_left_y in range(0, image.shape[0],stride):
        for top_left_x in range(0, image.shape[1],stride):
            confidence = np.array(predictions[prediction_index])
            sensitivity_map[
                top_left_y:top_left_y+patch_size,
                top_left_x:top_left_x+patch_size,
            ] = np.max(confidence)
            prediction_index += 1
    return sensitivity_map

def occlusion_sensitivity(
    model: tf.keras.Model,
    data: tf.data.Dataset,
    num_items: int,
    patch_size: int = 3,
    stride: int = 1):
    """
    Compute sensitivity to a series of data points by occluding a section of the image
    Occlusion is shifted across the image, with loss being computed at each step

    Parameters:
        model: tf.keras.Model
            The model to test sensitivity of
        data: tf.data.Dataset
            The data to test over. Should be a subsection of the entire dataset
            Use data.take() to get a small subset of data first
            Each image is processed, so the size of this dataset is number of images
        num_items: int
            The number of items to process overall (from the dataset)
        patch_size: int
            The size of the patch use for occlusion
        stride: int
            The amount to move the patch between trials
            Currently unused (stride = patch_size)
    """

    processed_images = 0

    # Loop over each batch in the dataset
    for batch in data:
        # Pull out the image/label pair from tuple
        # Each of these variables have dimension starting with batch_size
        # So zip them together to get image, label pairs directly
        images = batch[0]
        labels = batch[1]
        sensitivity_maps = []
        for image, label in zip(images, labels):  # type: ignore
            sensitivity_map = _process_occlude_image(model, image, label, patch_size, stride)
            sensitivity_maps.append(sensitivity_map)
            # Finally finished with one image! Check if we have met quota
            processed_images += 1
            if processed_images >= num_items:
                break
        plot_images(images[:num_items])
        plot_images(sensitivity_maps)
        # TODO Plot sensitivity maps returned and combine with original image...