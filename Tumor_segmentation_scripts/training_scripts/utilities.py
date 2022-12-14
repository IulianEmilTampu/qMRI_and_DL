import glob
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

matplotlib.use("pdf")
import nibabel as nib

# for opening nifti files
import random

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K


def tictoc(tic=0, toc=1):
    """
    # Returns a string that contains the number of days, hours, minutes and
    seconds elapsed between tic and toc
    """
    elapsed = toc - tic
    days, rem = np.divmod(elapsed, 86400)
    hours, rem = np.divmod(rem, 3600)
    minutes, rem = np.divmod(rem, 60)
    seconds = rem

    # form a string in the format d:h:m:s
    # return str(days)+delimiter+str(hours)+delimiter+str(minutes)+delimiter+str(round(seconds,0))
    return "%2dd:%02dh:%02dm:%02ds" % (days, hours, minutes, seconds)


# ################################## DATASET UTILITIES


def load_MR_modality(data, nr_to_load=0):

    """
    Loads and preproceses a the nii.gz files in the given data. Data can be
    eighter a folder or a list of files to process.

    INPUT
    - data: path to forlder or list pointing to the files to load
    - nr_to_load: number of cases in the folder to load (default = 0 -> all cases)

    RETURNS
    - images: a np array with shape [images, width, hight, 1]
    """

    image_archive = load_3D_data(data, nr_to_load)

    images = image_archive["data_volumes"][:, :, :, :, :]
    images = images.transpose(0, 3, 1, 2, 4)
    images = images.reshape(
        images.shape[0] * images.shape[1],
        images.shape[2],
        images.shape[3],
        images.shape[4],
    )

    images = data_normalization(images, quantile=0.995)

    return images


def data_normalization(dataset, type=1, quantile=0.995, clip=True, min=None, max=None):
    """
    Normalizes the data between [-1,1] using the specified quantile value.
    Note that the data that falls outside the [-1, 1] interval is not clipped,
    thus there will be values outside the normalization interval

    Using the formula:

        x_norm = 2*(x - x_min)/(x_max - x_min) - 1

    where
    x_norm:     normalized data in the [-1, 1] interval (note outliers outside the range)
    x:          data to normalize
    x_min:      min value based on the specified quantile
    x_max:      max value based on the specified quantile

    Args.
    dataset: numpy array to normalize
    type: 1 identifies [-1, 1] normalization,
    type: 2 identifies[0, 1] normalization,
    type: 3 identifies [-1, 1] using given min and max values
    type: 4 identifies [0, 1] using given min and max values

    quantile: value to use to find the min and the max (type 1 and 2)
    clip: if True, values above or below the min or max value will be clipped to
            min and max respectively.
    """
    if type == 1 or type == 2:
        min = np.quantile(dataset, 1 - quantile)
        max = np.quantile(dataset, quantile)

    if type == 1 or type == 3:
        norm = 2.0 * (dataset.astype("float32") - min) / (max - min) - 1.0
    elif type == 2 or type == 4:
        norm = (dataset.astype("float32") - min) / (max - min)

    if clip == True:
        if type == 1 or type == 3:
            norm[norm < -1] = -1
            norm[norm > 1] = 1
        if type == 2 or type == 4:
            norm[norm < 0] = 0
            norm[norm > 1] = 1

    return norm


def create_data_gen(X, Y, batch_size=32, seed=None):
    data_gen_args = dict(
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    if not seed:
        seed = np.random.randint(123456789)

    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(Y, batch_size=batch_size, seed=seed)

    # return image_generator, mask_generator
    return zip(image_generator, mask_generator)


def shuffle_array(x, y):
    """
    Utilitiy that given the image array and the mask array in using channel last
    convention [img, width, hight, channel], shuffles the arrays in the same way
    along the first dimension.
    """

    shuffled_indexes = random.sample(range(x.shape[0]), x.shape[0])
    aus_x = [x[i, :, :, :] for i in shuffled_indexes]
    aus_y = [y[i, :, :, :] for i in shuffled_indexes]
    return np.array(aus_x), np.array(aus_y)


def classWeights(Y):
    """
    Returns the normalized class weights for the classes in the cathegorical Y
    """
    num = len(Y.flatten())
    den = np.sum(Y, axis=tuple(range(Y.ndim - 1)))
    class_weights = np.square(num / den)
    return class_weights / np.sum(class_weights)


# ################################## ADJUST TUMOR ANNOTATION UTIITIES


def get_tumor_border(gt, n_border_pixels=2):
    from scipy import ndimage

    """
    Utility that given a binary mask identifying the tumor, returns a binary mask 
    identifying the tumor border identified as n_border_pixels arround the given 
    mask.
    
    Input
    gt : np array
        Binary mask of the tumor expected to be [width, hight]
    n_border_pixels : int
        Number of pixels that the border will be made of
    
    Returns
    border : np array
        Binary mask identifing the tumor border with widht n_border_pixels
    """

    # check that the input does have tumor in in
    if gt.sum() != 0:
        return (
            ndimage.binary_dilation(gt, iterations=n_border_pixels).astype(gt.dtype)
            - gt
        )
    else:
        # just return gt
        return gt


def adjust_tumor(gt, n_pixels=4):
    from scipy import ndimage

    """
    Utility that given a binary mask identifying the tumor, returns a binary mask 
    identifying the tumor erored or expanded based on the n_pixelss
    
    Input
    gt : np array
        Binary mask of the tumor expected to be [width, hight]
    n_pixels : int
        Number of pixels to add around the tumor mask (if positive) or remove
        if negative
    
    Returns
    border : np array
        Binary mask identifing adjusted tumor mask
    """

    # check that the input does have tumor in in
    if gt.sum() != 0:
        if n_pixels >= 0:
            return ndimage.binary_dilation(gt, iterations=n_pixels).astype(gt.dtype)
        elif n_pixels < 0:
            aus = ndimage.binary_erosion(gt, iterations=-n_pixels).astype(gt.dtype)
            # print('Inside function ' ,aus.sum())
            return aus
            # ndimage.binary_erosion(gt, iterations=-n_pixels).astype(gt.dtype)
    else:
        # just return gt
        return gt


# ################################## PLOTTING UTILITIES


def inspectDataset(volume_list, volume_GT=[], start_slice=0, end_slice=1, title=[]):
    """
    Creats an interactive window where one can move throught the images selected
    in the start and end_slice value.

    INPUT
    volume_list: list of volumes to display. These volumes should have the same
            number of slices
    volume_GT: list of ground truth to displayed overlayed to the volumes provided
            in volume list. If one volume only is provided than this is used as
            overlay to all the volumes. If more than one volume is given, then
            these shoudl be as many as the volumes in volume_list.
    start_slice: from where start to visualize the volumes
    end slice: to which slice to visualize the volumes
    titel: list of names to display for each volume. If only one is provided, the
            same title will be displayed for all volumes. If more than one is
            provided, then these shuold be as many as the volume in volume_list
    """
    # check the inputs
    if volume_GT:
        GT_given = True
        if len(volume_GT) == 1:
            # only one GT is provided, use this as overlay to all the volumes
            volume_GT = [volume_GT[0] for i in range(len(volume_list))]
        elif len(volume_GT) != len(volume_list):
            raise TypeError(
                "Invalid list of GT volumes. Lenth of input GT should be 1 or {}. Given {}".format(
                    len(volume_list), len(volume_GT)
                )
            )
    else:
        GT_given = False

    if title:
        title_given = True
        if len(title) == 1:
            # only one GT is provided, use this as overlay to all the volumes
            title = [title[0] for i in range(len(volume_list))]
        elif len(volume_GT) != len(volume_list):
            raise TypeError(
                "Invalid list of title. Lenth of input list should be 1 or {}. Given {}".format(
                    len(volume_list), len(title)
                )
            )
    else:
        title_given = False

    # create axisSequence object with as many axes as the number of volumes given
    axes = axesSequence(len(volume_list))

    # fill the axis
    for i, axs in zip(range(0, end_slice - start_slice), axes):
        sample = i + start_slice
        for j, ax in enumerate(axs.flat):
            if j < len(volume_list):
                # display volume
                V = volume_list[j]
                ax.imshow(
                    V[sample, :, :, 0],
                    cmap="gray",
                    interpolation="none",
                    vmin=V.min(),
                    vmax=V.max(),
                )
                if title_given:
                    ax.set_title(title[j])
                ax.set_xticks([])
                ax.set_yticks([])
                # overlay GT if given
                if GT_given:
                    GT = volume_GT[j]
                    for x in range(0, GT.shape[-1]):
                        ax.imshow(
                            np.ma.masked_where(
                                (x + 1) * GT[sample, :, :, x] <= 0.5,
                                (x + 1) * GT[sample, :, :, x],
                            ),
                            cmap="Set1",
                            norm=colors.Normalize(vmin=0, vmax=10),
                            alpha=0.5,
                        )
    axes.remove_unused_axes()
    axes.show()


class axesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""

    def __init__(self, n_axis=1):
        self.n_axis = n_axis
        if self.n_axis <= 3:
            self.n_cols = int(self.n_axis)
        else:
            self.n_cols = int(3)
        self.n_rows = int(np.ceil(n_axis / 3))
        self.fig = plt.figure(figsize=(10, 10))
        self.axes = []
        self._i = 0  # Currently displayed axes index
        self._n = 0  # Last created axes index
        self.fig.canvas.mpl_connect("key_press_event", self.on_keypress)

    def __iter__(self):
        while True:
            yield self.new()

    def new(self):
        # The label needs to be specified so that a new axes will be created
        # instead of "add_axes" just returning the original one.
        axs = self.fig.subplots(self.n_rows, self.n_cols, squeeze=0)
        # turn off all axes visibility
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                axs[r][c].set_visible(False)
                axs[r][c].set_label(self._n)
        self._n += 1
        self.axes.append(axs)
        return axs

    def on_keypress(self, event):
        if event.key == "right":
            self.next_plot()
        elif event.key == "left":
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()

    def next_plot(self):
        if self._i < len(self.axes) - 1:
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    self.axes[self._i][r][c].set_visible(False)
                    self.axes[self._i + 1][r][c].set_visible(True)
            self._i += 1
        else:
            print("No more slices")

    def prev_plot(self):
        if self._i > 0:
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    self.axes[self._i][r][c].set_visible(False)
                    self.axes[self._i - 1][r][c].set_visible(True)
            self._i -= 1
        else:
            print("No more slices")

    def show(self):
        # show the first set of axes -> set visibility to True
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                self.axes[0][r][c].set_visible(True)
        # turn off unused axes
        # self.remove_unused_axes()
        # plt.tight_layout()
        plt.show()

    def remove_unused_axes(self):
        # turn off unused axis
        if self.n_axis % 3 != 0:
            for i in range((self.n_axis % 3) - 1):
                self.axes[0][-1][-i].set_visible(False)


def plotEpochSegmentation(
    x, y, yPred, epoch=None, save_path=None, display=False, n_subjects_to_show=1
):
    """
    Plots the segmetnation of the model for three random samples in the given data.

    Inputs are expected to be numpy arrays.
    """
    # # supress wartning from numpy
    # warnings.simplefilter("ignore")

    # create image using two, one with
    slices_with_tumor = np.argwhere(y[:, :, :, 1].sum(axis=(1, 2)) != 0).flatten()
    if slices_with_tumor.size == 0:
        slices_with_tumor = np.floor(np.linspace(0, y.shape[0], y.shape[0] - 1)).astype(
            int
        )

    slices_without_tumor = np.argwhere(y[:, :, :, 1].sum(axis=(1, 2)) == 0).flatten()
    if slices_without_tumor.size == 0:
        slices_without_tumor = np.floor(
            np.linspace(0, y.shape[0], y.shape[0] - 1)
        ).astype(int)

    # # handle when there are no slices with or without tumor
    # if not any([slices_with_tumor.size==0, slices_without_tumor.size==0]):
    #     if slices_with_tumor.size==0:
    #         slices_with_tumor = slices_without_tumor
    #     if slices_without_tumor.size==0:
    #         slices_without_tumor = slices_with_tumor
    #     if slices_with_tumor.size==0 and slices_without_tumor.size==0:
    #         slices_with_tumor = np.linspace(0, y,shape[0], y,shape[0])
    #         slices_without_tumor = np.linspace(0, y,shape[0], y,shape[0])

    selected_w_t = np.random.choice(slices_with_tumor, size=n_subjects_to_show)
    selected_wo_t = np.random.choice(slices_without_tumor, size=n_subjects_to_show)

    fig, axs = plt.subplots(
        nrows=n_subjects_to_show * 2,
        ncols=y.shape[-1] * 2 + 1,
        figsize=(25, 10),
        gridspec_kw={"hspace": 0.1, "wspace": 0.2},
    )
    color_map = plt.cm.get_cmap("Set1").reversed()

    for ax, sw, swo in zip(
        range(0, n_subjects_to_show * 2, 2), selected_w_t, selected_wo_t
    ):
        # print image with tumor
        axs[ax][0].imshow(np.squeeze(x[sw, :, :, 0]), cmap="gray", interpolation=None)
        axs[ax][0].set_title("Raw image")

        # print subsequent y and yPred volumes
        for idx, g, t in zip(range(2), [y, yPred], ["GT", "Prediction"]):
            for j in range(g.shape[-1]):
                # print the overlayed segmentation/annotation
                axs[ax][j + idx * g.shape[-1] + 1].imshow(
                    np.squeeze(x[sw, :, :, 0]), cmap="gray", interpolation=None
                )
                # axs[ax+1][j+idx*g.shape[-1]+1].imshow(np.ma.masked_where(np.squeeze(g[sw,:,:,j]) <= 0.5, np.squeeze(g[sw,:,:,j])),
                #                             cmap = 'Set1',
                #                             alpha=0.5)
                axs[ax][j + idx * g.shape[-1] + 1].imshow(
                    np.squeeze(g[sw, :, :, j]),
                    vmin=0,
                    vmax=1,
                    cmap=color_map,
                    alpha=0.5,
                )
                axs[ax][j + idx * g.shape[-1] + 1].set_title(f"{t}-ch {j+1}")

        # print image without tumor
        axs[ax + 1][0].imshow(
            np.squeeze(x[swo, :, :, 0]), cmap="gray", interpolation=None
        )
        axs[ax + 1][0].set_title("Raw image")

        # print subsequent y and yPred volumes
        for idx, g, t in zip(range(2), [y, yPred], ["GT", "Prediction"]):
            for j in range(g.shape[-1]):
                # print the overlayed segmentation/annotation
                axs[ax + 1][j + idx * g.shape[-1] + 1].imshow(
                    np.squeeze(x[swo, :, :, 0]), cmap="gray", interpolation=None
                )
                # axs[ax+1][j+idx*g.shape[-1]+1].imshow(np.ma.masked_where(np.squeeze(g[swo,:,:,j]) <= 0.5, np.squeeze(g[swo,:,:,j])),
                #                             cmap = 'Set1',
                #                             alpha=0.5)
                axs[ax + 1][j + idx * g.shape[-1] + 1].imshow(
                    np.squeeze(g[swo, :, :, j]),
                    vmin=0,
                    vmax=1,
                    cmap=color_map,
                    alpha=0.5,
                )
                axs[ax + 1][j + idx * g.shape[-1] + 1].set_title(f"{t}-ch {j+1}")

    # remove ticks
    for ax in axs.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
    # reduce space
    # fig.subplots_adjust(wspace=-0.2)
    if save_path is not None:
        # check that path exist, if not Print error
        if os.path.isdir(save_path):
            fig.savefig(
                os.path.join(save_path, f"segmentation_validation_{epoch}.png"),
                bbox_inches="tight",
                dpi=100,
            )
        else:
            print(
                f"Given save folder does not exist. Provide a valid one. Given {save_path}"
            )
    if display is True:
        plt.show()
    else:
        plt.close()


def plotModelPerformance(
    tr_loss,
    tr_acc,
    val_loss,
    val_acc,
    tr_dice,
    val_dice,
    save_path,
    display=False,
    best_epoch=None,
):
    """
    Saves training and validation curves.
    INPUTS
    - tr_loss: training loss history
    - tr_acc: training accuracy history
    - tr_dice : training dice-score history
    - val_loss: validation loss history
    - val_acc: validation accuracy history
    - val_dice : validation dice-score history
    - save_path: path to where to save the model
    """

    fig, ax1 = plt.subplots(figsize=(15, 10))
    colors = [
        "blue",
        "orange",
        "green",
        "red",
        "pink",
        "gray",
        "purple",
        "brown",
        "olive",
        "cyan",
        "teal",
    ]
    line_style = [":", "-.", "--", "-"]
    ax1.set_xlabel("Epochs", fontsize=15)
    ax1.set_ylabel("Loss", fontsize=15)
    l1 = ax1.plot(tr_loss, colors[0], ls=line_style[2])
    l2 = ax1.plot(val_loss, colors[1], ls=line_style[3])
    plt.legend(["Training loss", "Validation loss"])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel("Accuracy and F1-score", fontsize=15)
    ax2.set_ylim(bottom=0, top=1)
    l3 = ax2.plot(tr_acc, colors[2], ls=line_style[2])
    l4 = ax2.plot(val_acc, colors[3], ls=line_style[3])
    l5 = ax2.plot(tr_dice, colors[4], ls=line_style[2])
    l6 = ax2.plot(val_dice, colors[5], ls=line_style[3])
    if best_epoch:
        l7 = ax2.axvline(x=best_epoch)

    # add legend
    if best_epoch:
        lns = l1 + l2 + l3 + l4 + l5 + l6 + l7
        labs = [
            "Training loss",
            "Validation loss",
            "Training accuracy",
            "Validation accuracy",
            "Training Dice-score",
            "Validation Dice-score",
            "Best_model",
        ]
        ax1.legend(lns, labs, loc=7, fontsize=15)
    else:
        lns = l1 + l2 + l3 + l4 + l5 + l6
        labs = [
            "Training loss",
            "Validation loss",
            "Training accuracy",
            "Validation accuracy",
            "Training Dice-score",
            "Validation Dice-score",
        ]
        ax1.legend(lns, labs, loc=7, fontsize=15)

    ax1.set_title("Training loss, accuracy and Dice-score trends", fontsize=20)
    ax1.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(os.path.join(save_path, "perfomance.pdf"), bbox_inches="tight", dpi=100)
    fig.savefig(os.path.join(save_path, "perfomance.png"), bbox_inches="tight", dpi=100)
    plt.close()

    if display is True:
        plt.show()
    else:
        plt.close()


def plotResultsPrediction(
    Raw_image,
    GT,
    Prediction,
    focusClass,
    nExamples=1,
    randomExamples=False,
    save_figure=False,
    save_path="",
    raw_data_channel=0,
):
    """
    plotResultsPrediction creates a figure that shows the comparison between the
    GT and the network Prediction. It focuses the attention on the specified
    focusClass such that nExamples of GT with high, medium and low number of
    focusClass pixels/voxels are presented. This in order to show the performance
    of the network in different shenarious.
                            [INPUTS]
    - Raw_image: input images used for the training of the network
    - GT: ground truth (cathegorical)
    - Prediction: network prediction
    - focusClass: class or classes that one is more interested in seeing the
                  performance of the network
    - nExamples: number of examples for high, medium and low number of focusClass
                 pixels/voxels
    - save_figure = if one wants to save the figure
    - save_path = where to save the figure
    - raw_data_channel = what channel of the Raw_data used for the plot

                                [ALGORITHM]
    1 - for all the samples in GT, sum along focusClass
    2 - order the obtained sum from the smallest to the largest, keeping the track
        of the indexes of the sample the sum belongs to
    3 - based on the number of samples, devide the ordered sum in three parts (low,
       medium and high number of focusClass pixels/voxels)
    4 - take nExamples random samples from the three different parts
    5 - plot the selected samples
        - first row: raw image
        - second row: raw image with superimposed GT
        - third row: raw image with superimposed Prediction
    6 - save the image is required
    """

    from random import randint
    import matplotlib.colors as colors
    import matplotlib.gridspec as gridspec

    # 1- for all the samples, sum along the focusClasses
    focusClassSum = np.sum(
        np.take(GT, focusClass, axis=GT.ndim - 1), axis=tuple(range(1, GT.ndim))
    )

    # 2 - order keeping track of the indexes
    orderedIndexes = np.argsort(focusClassSum)
    # print(orderedIndexes)

    # 3/4 - take out nExamples random from three different parts of the orderedIndexes
    nSp = np.floor_divide(orderedIndexes.shape[0], 3)
    # if random is selected
    if randomExamples == True:
        randomSamples = np.array(
            [
                np.random.choice(orderedIndexes[0 : nSp - 1], 3),
                np.random.choice(orderedIndexes[nSp : nSp * 2 - 1], 3),
                np.random.choice(orderedIndexes[nSp * 2 : nSp * 3 - 1], 3),
            ]
        ).reshape(-1)

    randomSamples = np.array(
        [
            orderedIndexes[0:nExamples],
            orderedIndexes[
                nSp + int(round(nSp / 2, 0)) : nSp + int(round(nSp / 2, 0)) + nExamples
            ],
            orderedIndexes[nSp * 3 - nExamples - 1 : nSp * 3 - 1],
        ]
    ).reshape(-1)
    # print(randomSamples)

    # 5 - plot selected samples - FOR NOW THIS WORKS FOR 2D DATASETS NOT 3D
    fig = plt.figure(figsize=(4, 4))
    plt.suptitle("Examples of predictions", fontsize=20)
    nfr = nExamples * 3  # numer of total samples per row
    for i in range(nfr):
        plt.subplot(3, nfr, i + 1)
        if i == 0:
            plt.ylabel("Original Image", fontsize=15)
            # code here for the raw image
        plt.imshow(
            Raw_image[randomSamples[i], :, :, raw_data_channel],
            cmap="gray",
            interpolation="none",
        )
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3, nfr, nfr + i + 1)
        if nfr + i == nfr:
            plt.ylabel("Ground Truth", fontsize=15)
            # code here to show the raw image with superimposed all the different classes
        plt.imshow(
            Raw_image[randomSamples[i], :, :, raw_data_channel],
            cmap="gray",
            interpolation="none",
        )
        for j in range(1, GT.shape[-1]):
            plt.imshow(
                np.ma.masked_where(
                    j * GT[randomSamples[i], :, :, j] == 0,
                    j * GT[randomSamples[i], :, :, j],
                ),
                cmap="Dark2",
                norm=colors.Normalize(vmin=0, vmax=GT.shape[-1]),
            )
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3, nfr, nfr * 2 + i + 1)
        if 2 * nfr + i == 2 * nfr:
            plt.ylabel("Prediction", fontsize=15)
        # code here to show raw image with superimposed all the predicted classes
        plt.imshow(
            Raw_image[randomSamples[i], :, :, raw_data_channel],
            cmap="gray",
            interpolation="none",
        )
        for j in range(1, GT.shape[-1]):
            plt.imshow(
                np.ma.masked_where(
                    j * Prediction[randomSamples[i], :, :, j] <= 0.1,
                    j * Prediction[randomSamples[i], :, :, j],
                ),
                cmap="Dark2",
                norm=colors.Normalize(vmin=0, vmax=GT.shape[-1]),
            )
        plt.xticks([])
        plt.yticks([])
    fig.tight_layout()
    fig.show()

    if save_figure == True:
        fig.savefig(os.path.join(save_path, "predictionExamples.pdf"))


# ################################## OTHER UTILITIES


def tictoc_from_time(elapsed=1):
    """
    # Returns a string that contains the number of days, hours, minutes and
    seconds given the elapsed time
    """
    days, rem = np.divmod(elapsed, 86400)
    hours, rem = np.divmod(rem, 3600)
    minutes, rem = np.divmod(rem, 60)
    seconds, rem = np.divmod(rem, 1)
    milliseconds = rem * 1000

    # form a string in the format d:h:m:s
    # return str(days)+delimiter+str(hours)+delimiter+str(minutes)+delimiter+str(round(seconds,0))
    return "%2dd:%02dh:%02dm:%02ds:%02dms" % (
        days,
        hours,
        minutes,
        seconds,
        milliseconds,
    )
