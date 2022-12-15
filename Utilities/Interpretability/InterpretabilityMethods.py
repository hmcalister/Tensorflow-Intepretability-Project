# fmt: off
from typing import List, Tuple, Union
from Utilities.Utils import plot_images
import numpy as np
import matplotlib.pyplot as plt
import cv2

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# fmt: on

def find_last_conv_layer(model:tf.keras.models.Model) -> tf.keras.layers.Conv2D:
    last_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
        if isinstance(layer, tf.keras.models.Model):
            submodel_last_conv_layer = find_last_conv_layer(layer)
            if submodel_last_conv_layer is not None:
                last_conv_layer = submodel_last_conv_layer
    return last_conv_layer

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
def _process_filter(
        submodel: tf.keras.models.Model,
        filter_index: int,
        steps: int,
        step_size: float
    ):
    # Get input shape and replace None (batch placeholder) with 1
    noise_shape = submodel.input_shape
    noise_shape = (1, *noise_shape[1:])
    x: tf.Variable = tf.Variable(tf.cast(np.random.random(noise_shape), tf.float32))
    for _ in range(steps):
        with tf.GradientTape() as tape:
            y: tf.Tensor = submodel(x)  # type: ignore
            loss = tf.reduce_mean(y[:,:,:,filter_index])  # type: ignore
        grads = tape.gradient(loss, x)
        normalized_grads = grads/(tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
        x.assign_add( normalized_grads*step_size )
    # Rescale float to be between 0,1
    x = (x-tf.math.reduce_min(x)) / (tf.math.reduce_max(x) - tf.math.reduce_min(x))
    return x
    
def kernel_activations(
        model: tf.keras.Model,
        layer_name: str = None,  # type: ignore
        steps:int = 500,
        step_size = 0.01,
        cmap:str = None,  # type: ignore
        filters_per_plot:int=None, #type: ignore
        save_path: str | None= None,
        **kwargs
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
        cmap: str
            The color map to use for image plotting
            Defaults to None
        max_filters: int
            The maximum number of filters to display in one plot
            If 0 (default) display all filters
            Notice last plot will have remainder of filters
            (probably fewer than max_filters)
        save_path: str | None
            If present plots are saved (not shown) to the path
            Note save_path should point to a directory into which
            figures will be saved

    """

    # Target layer of kernel inspection, either user defined or last conv layer
    target_layer: tf.keras.layers.Layer
    if layer_name is not None:
       target_layer = model.get_layer(layer_name)
    # if no layer name specified, just get last conv layer by iterating over all layers
    if layer_name is None:
        target_layer = find_last_conv_layer(model)
        layer_name = target_layer.name
    print(f"OPERATING ON {target_layer.name}")

    submodel = tf.keras.models.Model(
        model.input,
        [target_layer.output]
    )

    if filters_per_plot is None:
        filters_per_plot = target_layer.filters

    filter_results = []
    # Get input shape and replace None (batch placeholder) with 1
    noise_shape = submodel.input_shape
    noise_shape = (1, *noise_shape[1:])
    for filter_index in range(1, target_layer.filters+1):
        print(f"OPERATING ON FILTER: {filter_index}/{target_layer.filters}", end="\r")
        x = _process_filter(submodel, filter_index-1, steps, step_size)
        filter_results.append(x[0,:,:,:])  # type: ignore

        if filter_index % filters_per_plot == 0 or filter_index == target_layer.filters:
            if save_path is not None:
                im_path = f"{save_path}/{target_layer.name}-KernelActivation{filter_index-len(filter_results)}-{filter_index}.png"
            else: 
                im_path = None
            plot_images(filter_results, 
                figure_title=f"Kernel Activations\n{model.name} - {layer_name}",
                subplot_titles=[f"Filter {i+1}" for i in range(filter_index-len(filter_results),filter_index)],
                cmap=cmap,
                save_plot=im_path,
                **kwargs
            )

            filter_results = []
            # Get input shape and replace None (batch placeholder) with 1
            noise_shape = submodel.input_shape
            noise_shape = (1, *noise_shape[1:])
    print()

def _process_occlude_image(
    model: tf.keras.Model,
    image: np.ndarray,
    label: np.ndarray,
    patch_size: int,
    stride: int,
    batch_size: int
):

    sensitivity_map = np.zeros((image.shape[0], image.shape[1]))
    patched_images = []
    patch_value = np.average([np.max(image), np.min(image)])
    top_left_y = 0
    top_left_x = 0
    patch_index = 0
    prediction_index = 0

    while top_left_y < image.shape[0] - patch_size or top_left_x < image.shape[1]-patch_size:
        top_left_y = patch_index*stride // (image.shape[0] - patch_size + 1)
        top_left_x = patch_index*stride % (image.shape[1] - patch_size + 1)
        print(f"PATCHES PROCESSED {patch_index+1} {top_left_x, top_left_y}", end="\r")
        # Copy the original image, apply a square of black over that patch
        patched_image = np.array(image, copy=True)
        patched_image[
            top_left_y:top_left_y+patch_size,
            top_left_x:top_left_x+patch_size, 
            : 
        ] = patch_value

        patch_index += 1
        # We collect all the patched images together to be processed in a single batch
        patched_images.append(patched_image)
        if len(patched_images) >= batch_size:
            patched_images = np.array(patched_images)
            # Note multiplying by label, we are only interested in 
            # model predicted likelihood of correct class
            predictions = model(patched_images) * label # type: ignore
            # Loop over predictions and apply confidence to the sensitivity map
            for prediction in predictions:
                sensitivity_top_left_y = prediction_index*stride // (image.shape[0] - patch_size + 1)
                sensitivity_top_left_x = prediction_index*stride % (image.shape[1] - patch_size + 1)
                confidence = np.array(prediction)
                sensitivity_map[
                    sensitivity_top_left_y:sensitivity_top_left_y+patch_size,
                    sensitivity_top_left_x:sensitivity_top_left_x+patch_size,
                ] = np.max(confidence)
                prediction_index += 1
            patched_images = []
    patched_images = np.array(patched_images)
    # Note multiplying by label, we are only interested in 
    # model predicted likelihood of correct class
    predictions = model(patched_images) * label # type: ignore
    # Loop over predictions and apply confidence to the sensitivity map
    for prediction in predictions:
        sensitivity_top_left_y = prediction_index*stride // (image.shape[0] - patch_size + 1)
        sensitivity_top_left_x = prediction_index*stride % (image.shape[1] - patch_size + 1)
        confidence = np.array(prediction)
        sensitivity_map[
            sensitivity_top_left_y:sensitivity_top_left_y+patch_size,
            sensitivity_top_left_x:sensitivity_top_left_x+patch_size,
        ] = np.max(confidence)
        prediction_index += 1
    print()
    return sensitivity_map

def occlusion_sensitivity(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    patch_size: int = 3,
    stride: int = 1,
    occlusion_batch_size: int = 32):
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
        batch_size: int
            number of patched images to process at once

    """

    sensitivity_maps = []
    for i, (image, label) in enumerate(zip(images, labels)):
        print(f"IMAGE {i+1}/{len(images)}")
        sensitivity_map = _process_occlude_image(model, image, label, patch_size, stride, occlusion_batch_size)
        sensitivity_maps.append(sensitivity_map)
    # plot_images(images[:num_items])
    plot_images(sensitivity_maps, figure_title=f"Occlusion Sensitivity - {model.name}")
    # TODO Plot sensitivity maps returned and combine with original image...

def GRADCAM(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    layer_name:str = None, # type: ignore
    alpha: float = 0.5,
    beta: float = 0.5,
    ignore_negative_gradients: bool = False,
    show_predictions: int = 0,
    absolute_scale: Union[float, None] = None,): 
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
        alpha: float
            The amount of original image to display (between 0,1)
        beta: float
            The amount of heatmap to overlay (between 0,1)
        ignore_negative_gradients: bool
            Boolean to ignore negative gradients
            May give better results but may have no good basis in theory
        show_predictions: int
            Show model predictions of classes (and true classes) as subplot titles
            Given integer is number of decimal points to round softmax model output to
            If given 0 (default) titles are eschewed
        absolute_scale: Union[float, None]
            Float value to use for scaling the heatmap
            If None (default) then heat map is scaled automatically to fit between 0,1
            Useful to ensure tiny relative differences are presented on same scale as 
            large relative differences
    """

    # The input layer of the gradcam model is the input of the original model
    original_inputs = model.inputs
    # We also need the original model output
    original_output = model.output
    # the gradcam output is trickier, either specified layer or final conv layer
    gradcam_output_layer: tf.keras.layers.Layer = None # type: ignore
    if layer_name is not None:
       gradcam_output_layer = model.get_layer(layer_name)
    # if no layer name specified, just get last conv layer by iterating over all layers
    else:
        gradcam_output_layer = find_last_conv_layer(model)
    assert gradcam_output_layer is not None 
    
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
    heatmax_ranges = []
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
        if absolute_scale != None:
            scale = absolute_scale
        else:
            scale = (cam.max() - cam.min())
        heatmap = (cam - cam.min()) / scale
        heatmax_ranges.append((cam.max() - cam.min()))
        cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)  # type: ignore
        # Handle conversion to color image if not already
        image = np.uint8(255*image)
        if image.shape[-1] == 1:  # type: ignore
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # type: ignore
        gradcam_image = cv2.addWeighted(image, alpha, cam, beta, 0)  # type: ignore
        gradcam_images.append(gradcam_image)
    
    print(f"HEATMAP RANGE (EXTREME): {np.max(heatmax_ranges)}")
    print(f"HEATMAP RANGE (AVG): {np.average(heatmax_ranges)}")
    subplot_titles = []
    if show_predictions > 0:
        subplot_titles = [
            f"Label: {l}\nPrediction\n{np.around(tf.nn.softmax(p), show_predictions)}" for l, p in zip(labels, predictions)
        ]
    plot_images(gradcam_images, figure_title=f"GRADCAM\n{model.name} - {layer_name}", subplot_titles=subplot_titles)
