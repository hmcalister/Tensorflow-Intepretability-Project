# fmt: off
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

def plot_images(images: List[np.ndarray], figure_title:str = "", subplot_titles:List[str]=[], cmap: str = "viridis"):
    """
    Plot a series of images in a grid using matplotlib and subplots
    Useful to show interpretations of different filters in conv-nets

    Parameters:
        images: List[np.ndarray]
            The list of images to be plotted
            Images (items of the array) should be square, etc.. and ar passed directly to imshow
            The list itself can be any length (even non-square)

        figure_title: str
            The title of the entire figure

        subplot_titles: List[str]
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
    fig.suptitle(figure_title)
    for i in range(0,len(images)):
        ax = fig.add_subplot(total_rows, total_cols, i+1)
        ax.imshow(images[i], cmap=cmap)
        ax.axis("off")
        if i < len(subplot_titles):
            ax.set_title(subplot_titles[i], fontsize=8)
    plt.tight_layout()
    plt.show()

def kernel_inspection(
        model: tf.keras.Model,
        layer_name: str = None,  # type: ignore
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
        layer_name: str
            The layer from the model to use for kernel inspection
            If layer_name is None (default) then use final conv2D layer instead
        steps: int
            The number of steps to iterate over when building a visualization
            More steps gives a smoother/better interpretation
        step_size: float
            The influence of each step on the visualisation
            Smaller steps results in longer wait times but more stable visuals
    """

    # Target layer of kernel inspection, either user defined or last conv layer
    target_layer: tf.keras.layers.Layer
    if layer_name is not None:
       target_layer = model.get_layer(layer_name)
    # if no layer name specified, just get last conv layer by iterating over all layers
    if layer_name is None:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
    print(f"OPERATING ON {target_layer.name}")

    submodel = tf.keras.models.Model(
        model.input,
        [target_layer.output]
    )

    filter_results = []
    # Get input shape and replace None (batch placeholder) with 1
    noise_shape = submodel.input_shape
    noise_shape = (1, *noise_shape[1:])
    for filter_index in range(target_layer.filters):
        print(f"OPERATING ON FILTER: {filter_index+1}/{target_layer.filters}", end="\r")
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
        figure_title=f"Kernel Inspection\n{model.name} - {layer_name}",
        subplot_titles=[f"Filter {i+1}" for i in range(0,len(filter_results))],
        cmap="gray"
    )

def _process_occlude_image(
    model: tf.keras.Model,
    image: np.ndarray,
    label: np.ndarray,
    patch_size: int,
    stride: int
):
    # Convert the image to something more... manageable
    # Remove first index (added from tf dataset)
    # convert to numpy array
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
    images: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 3,
    stride: int = 1):
    """
    Compute sensitivity to a series of data points by occluding a section of the image
    Occlusion is shifted across the image, with loss being computed at each step

    Parameters:
        model: tf.keras.Model
            The model to test sensitivity of
        images: np.ndarray
            An array of images to be processed in this method
            Can be taken from a dataset and stored 
            (e.g. images,labels = dataset.take(1).as_numpy_iterator().next())
            Should have shape (batch_size, image_x, image_y, color_channels)
        labels: np.ndarray
            An array of labels to be processed, one to one with images array
        num_items: int
            The number of items to process overall (from the dataset)
        patch_size: int
            The size of the patch use for occlusion
        stride: int
            The amount to move the patch between trials
            Currently unused (stride = patch_size)
    """

    sensitivity_maps = []
    for image, label in zip(images, labels):
        sensitivity_map = _process_occlude_image(model, image, label, patch_size, stride)
        sensitivity_maps.append(sensitivity_map)
    # plot_images(images[:num_items])
    plot_images(sensitivity_maps, figure_title=f"Sensitivity Mappings - {model.name}")
    # TODO Plot sensitivity maps returned and combine with original image...

def GRADCAM(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    layer_name:str = None, # type: ignore
    ignore_negative_gradients: bool = False,
    show_predictions: bool = False): 
    """
    Compute GRADCAM over the number of items required.
    Note that data should be a tf.data.Dataset so the class of each image
    can be extracted from the data.

    Parameters:
        model: tf.keras.Model
            The model to test sensitivity of
        images: np.ndarray
            An array of images to be processed in this method
            Can be taken from a dataset and stored 
            (e.g. images,labels = dataset.take(1).as_numpy_iterator().next())
            Should have shape (batch_size, image_x, image_y, color_channels)
        labels: np.ndarray
            An array of labels to be processed, one to one with images array
        layer_name: str
            The layer from the model to use for GRADCAM
            If layer_name is None (default) then use final conv2D layer instead
        ignore_negative_gradients: bool
            Boolean to ignore negative gradients
            May give better results but may have no good basis in theory
        show_predictions: bool
            Boolean to show predictions as title of subplots
    """

    # The input layer of the gradcam model is the input of the original model
    original_inputs = model.inputs
    # We also need the original model output
    original_output = model.output
    # the gradcam output is trickier, either specified layer or final conv layer
    gradcam_output_layer: tf.keras.layers.Layer
    if layer_name is not None:
       gradcam_output_layer = model.get_layer(layer_name)
    # if no layer name specified, just get last conv layer by iterating over all layers
    if layer_name is None:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                gradcam_output_layer = layer
    
    # Create the new gradcam model combining inputs
    gradcam_model = tf.keras.models.Model([original_inputs], [gradcam_output_layer.output, original_output])
    gradcam_images = []

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(images)  # type: ignore
        # Note multiplying labels by predictions gets us only the "correct" predictions
        # as labels are one-hot. This is the class-activated part
        loss = labels * predictions
    grads = tape.gradient(loss, conv_outputs)
    # Now we have the processed information we can work image by image
    for index, (image, _) in enumerate(zip(images, labels)):  # type: ignore
        # Output of last conv layer to be scaled by "importance" (gradient)
        output = conv_outputs[index]
        current_grads = grads[index]
        weights = tf.reduce_mean(current_grads, axis=(0,1))
        cam = np.ones(output.shape[0:2], dtype=np.float32)
        for index, w in enumerate(weights):
            cam += w * output[:, :, index]

        if ignore_negative_gradients:
            # Zero out negative grads
            cam = np.maximum(cam, 0)
        # Stretch last conv output to original image size
        cam = cv2.resize(np.array(cam), image.shape[0:2])
        # Do some image processing to place CAM as heatmap on top of original image
        heatmap = cv2.cvtColor((cam - cam.min()) / (cam.max() - cam.min()), cv2.COLOR_BGR2RGB)
        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)  # type: ignore
        color_image = cv2.cvtColor(np.uint8(255*image), cv2.COLOR_GRAY2RGB)  # type: ignore
        gradcam_image = cv2.addWeighted(color_image, 0.5, cam, 0.6, 0)
        gradcam_images.append(gradcam_image)
    
    subplot_titles = []
    if show_predictions:
        subplot_titles = [
            f"Label: {l}\nPrediction\n{np.around(tf.nn.softmax(p), 2)}" for l, p in zip(labels, predictions)
        ]
    plot_images(gradcam_images, figure_title="GRADCAM", subplot_titles=subplot_titles)    plot_images(gradcam_images, figure_title=f"GRADCAM\n{model.name} - {layer_name}", subplot_titles=subplot_titles)