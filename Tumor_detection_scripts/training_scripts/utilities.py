import os
import glob
import random
from random import randint
import numpy as np
import nibabel as nib

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# simple utilities
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


# ############################## DATA RELATED UTILITIES


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


def create_data_gen(
    X, Y, batch_size=32, seed=None, classification=False, test_set=False
):
    if test_set:
        data_gen_args = dict(
            rotation_range=0,
            width_shift_range=0,
            height_shift_range=0,
            horizontal_flip=False,
            vertical_flip=False,
            zoom_range=0,
        )
    else:
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

    if classification:
        # transform the mask generator to output 0 or 1 besed on if the slices have mask or not
        # mask_generator = mask_generator.map(lambda y: tf.keras.utils.to_categorical(tf.cast(tf.greater(tf.reduce_sum(y),0),int16),num_classes=2))
        mask_generator = map(
            lambda y: tf.keras.utils.to_categorical(
                tf.cast(
                    tf.greater(tf.reduce_sum(y[:, :, :, 1], axis=(1, 2)), 0), tf.int16
                ),
                num_classes=2,
            ),
            mask_generator,
        )

    # return image_generator, mask_generator
    return zip(image_generator, mask_generator)
    # datagen = ImageDataGenerator(**data_gen_args)
    # return datagen.flow(X,Y,batch_size=batch_size)


def create_data_gen_better(X, Y, batch_size=32, seed=None):
    import imgaug as ia
    import imgaug.augmenters as iaa

    def_alpha = (0, 2)
    def_sigma = (0, 2)

    # define a preprocessing functions fpr both images and masks
    def elastic_deformation_img(image):
        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        aug = iaa.ElasticTransformation(
            alpha=def_alpha, sigma=def_sigma, order=3, mode="reflect"
        )
        image = aug.augment_image(image)
        return image

    def elastic_deformation_seg(seg):
        seq = iaa.Sequential(
            [
                iaa.ElasticTransformation(
                    alpha=def_alpha, sigma=def_sigma, order=3, mode="reflect"
                )
            ]
        )

        seg = np.expand_dims(seg.astype(np.int32), axis=0)
        _, seg = seq(images=seg, segmentation_maps=seg)
        return seg

    # arguments for the tf data generator
    data_gen_args_img = dict(
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        preprocessing_function=elastic_deformation_img,
    )

    data_gen_args_seg = dict(
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        preprocessing_function=elastic_deformation_seg,
    )

    image_datagen = ImageDataGenerator(**data_gen_args_img)
    mask_datagen = ImageDataGenerator(**data_gen_args_seg)

    # Provide the same seed and keyword arguments to the fit and flow methods
    if not seed:
        seed = np.random.randint(123456789)

    image_generator = image_datagen.flow(X, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(Y, batch_size=batch_size, seed=seed)

    # return image_generator, mask_generator
    return zip(image_generator, mask_generator)
    # datagen = ImageDataGenerator(**data_gen_args)
    # return datagen.flow(X,Y,batch_size=batch_size)


def load_3D_data(data="", nr_to_load=0, load_dimension=None, load_view="t"):
    """
     loads all (if nr_to_load is not given) the nifti files in the data if data is
     a directory. Else, if data is amlist of files, loads all the specified files.
     The function returns a list containing the information about the opened data:
     number of volumes, volume sizes, the number of channels and the multidimensional
     array containing all the datasets in the data_directory (this list can be
     extended with new features if needed)

    load_view -> this allows me to choose if I concatenate the volumes from a coronal,
    sagittal or transversal point of view. 1 -> transversal, 2 -> coronal, 3 -> sagittal.
    Assuming that the volumes are RAI oriented, in (x,y,z)
    coordinates x-> range SAGITTAL slices, y -> range CORONAL slices, z -> range TRANSVERSAL slices
    """

    # check if data is a list or a string
    if isinstance(data, list):
        # check that all the files in the list exist
        if not all([os.path.isfile(x) for x in data]):
            raise ValueError(
                "Some of the given files to load do not exist. Check files"
            )
        else:
            # set volume_names and datas_directory to use the create_volume_array function
            data_directory = os.path.dirname(data[0])
            volume_names = [os.path.basename(x) for x in data]
            nr_to_load = len(volume_names)
    elif isinstance(data, list):
        # check if the data_directory exists
        if not os.path.isdir(data):
            raise ValueError(f"The dataset folder does not exist. Given {data}")
        else:
            # get file names based on the number of subjects to load
            data_directory = data
            volume_names = glob.glob(os.path.join(data_directory, "*.nii.gz"))

            # now just take the file names
            volume_names = [os.path.basename(x) for x in volume_names]

            # set number of volumes to open
            if nr_to_load == 0:
                # load all the volumes in the directory
                nr_to_load = len(volume_names)
    else:
        raise ValueError(f"Expected data to by list or string but given {type(data)}")

    # check one dataset. This will give the information about the size of the
    # volumes and the number of channels
    volume_test = nib.load(os.path.join(data_directory, volume_names[0]))

    if len(volume_test.shape) == 3:
        volume_size = volume_test.shape
        nr_of_channels = 1
    else:  # e.g. if the dataset were aquired in different MR modalities and saved together
        volume_size = volume_test.shape[
            0:-1
        ]  # from the first element till the last excluded
        nr_of_channels = volume_test.shape[-1]  # the last element of the array

    header_test = volume_test.get_header()

    # use utility function create_volume_array to create the multidimensional
    # array that contains all the data in the specified folder
    data_volumes = create_volume_array(
        data_directory,
        volume_names,
        volume_size,
        nr_of_channels,
        nr_to_load,
        load_dimension,
        load_view,
    )

    # return dictionary
    return {
        "volume_size": volume_size,
        "nr_of_channels": nr_of_channels,
        "header": header_test,
        "data_volumes": data_volumes,
        "volume_names": volume_names,
    }


def create_volume_array(
    data_directory,
    volume_names,
    volume_size,
    nr_of_channels,
    nr_to_load,
    load_dimension,
    load_view,
    verbose=False,
):
    if verbose == True:
        # set progress bar
        bar = Bar("Loading...", max=nr_to_load)

    # initialize volume array depending on the required dimentionality and view
    if load_dimension == None:
        data_array = np.empty(
            (nr_to_load,) + (volume_size) + (nr_of_channels,), dtype="float32"
        )
    elif load_dimension == 2:
        aus_s = nr_to_load
        aus_x = volume_size[0]
        aus_y = volume_size[1]
        aus_z = volume_size[2]
        aus_c = nr_of_channels

        if load_view.lower() == "t":  # TRANSVERSAL
            # here the size is [number of volumes*size in z, size in x, size in y, channels)
            data_array = np.empty((aus_s * aus_z, aus_x, aus_y, aus_c), dtype="float32")
        elif load_view.lower() == "c":  # CORONAL
            # here the size is [number of volumes*size in y, size in x, size in z, channels)
            data_array = np.empty((aus_s * aus_y, aus_x, aus_z, aus_c), dtype="float32")
        elif load_view.lower() == "s":  # SAGITTAL
            # here the size is [number of volumes*size in x, size in y, size in z, channels)
            data_array = np.empty((aus_s * aus_x, aus_y, aus_z, aus_c), dtype="float32")
        else:
            print("Invalid view code. Select between t, s and c")
            return

    i = 0  # to index data_array

    # open and save in data_array all the volumes in volume_names
    for volume_name in volume_names[0:nr_to_load]:
        # load and convert to np array
        volume = nib.load(
            os.path.join(data_directory, volume_name)
        ).get_fdata()  # note that data is [0,1] norm
        volume = volume.astype("float32")

        # add 3rd dimension if data is 2D
        if nr_of_channels == 1:
            volume = volume[:, :, :, np.newaxis]

        # add volume to array based on the specification
        if load_dimension == None:
            data_array[i, :, :, :, :] = volume
        elif load_dimension == 2:
            if load_view.lower() == "t":  # TRANSVERSAL
                data_array[i * aus_z : (i + 1) * aus_z] = volume.transpose(2, 0, 1, 3)
            elif load_view.lower() == "c":  # CORONAL
                data_array[i * aus_y : (i + 1) * aus_y] = volume.transpose(1, 0, 2, 3)
            elif load_view.lower() == "s":  # SAGITTAL
                data_array[i * aus_x : (i + 1) * aus_x] = volume

        i += 1
        if verbose == True:
            bar.next()

    if verbose == True:
        bar.finish()
    return data_array


## ### GENERAL DATA LOADING FUNCTION


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


# ############################## PLOTTING UTILITIES


def inspectDataset_v2(volume_list, volume_GT=[], start_slice=0, end_slice=1, title=[]):
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
    axes = axesSequence_v2(len(volume_list))

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


class axesSequence_v2(object):
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
        plt.tight_layout()
        plt.show()

    def remove_unused_axes(self):
        # turn off unused axis
        if self.n_axis % 3 != 0:
            for i in range((self.n_axis % 3) - 1):
                self.axes[0][-1][-i].set_visible(False)


##


def plotModelPerformance_v2(
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


# ############################## ANOTATION MOD UTILITIES


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


# ############################## EVALUATION UTILITIES


def get_performance_metrics(true_logits, pred_softmax, average="macro"):
    from sklearn.metrics import (
        average_precision_score,
        recall_score,
        roc_auc_score,
        f1_score,
        accuracy_score,
        matthews_corrcoef,
        confusion_matrix,
    )

    """
    Utility that returns the evaluation metrics as a disctionary.
    THe metrics that are returns are:
    - accuracy
    - f1-score
    - precision and recall
    - auc
    - MCC

    INPUT
    Ytest : np.array
        Array containing the ground truth for the test data
    Ptest_softmax : np.array
        Array containing the softmax output of the model for each
        test sample

    OUTPUT
    metrics_dict : dictionary
    """
    # compute confusion matrix
    cnf_matrix = confusion_matrix(
        np.argmax(true_logits, axis=-1), np.argmax(pred_softmax, axis=-1)
    )

    # get TP, TN, FP, FN
    FP = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)).astype(float)
    FN = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)).astype(float)
    TP = (np.diag(cnf_matrix)).astype(float)
    TN = (cnf_matrix.sum() - (FP + FN + TP)).astype(float)

    # compute class metrics
    summary_dict = {
        "precision": TN / (FP + TN),
        "recall": TP / (TP + FN),
        "accuracy": (TP + TN) / (TP + TN + FP + FN),
        "f1-score": TP / (TP + 0.5 * (FP + FN)),
        "auc": roc_auc_score(true_logits, pred_softmax, average=None),
    }

    # compute overall metrics
    summary_dict["overall_precision"] = average_precision_score(
        true_logits, pred_softmax, average=average
    )
    summary_dict["overall_recall"] = recall_score(
        np.argmax(true_logits, axis=-1),
        np.argmax(pred_softmax, axis=-1),
        average=average,
    )
    summary_dict["overall_accuracy"] = accuracy_score(
        np.argmax(true_logits, axis=-1),
        np.argmax(pred_softmax, axis=-1),
    )
    summary_dict["overall_f1-score"] = f1_score(
        np.argmax(true_logits, axis=-1),
        np.argmax(pred_softmax, axis=-1),
        average=average,
    )
    summary_dict["overall_auc"] = roc_auc_score(
        true_logits,
        pred_softmax,
        average=average,
        multi_class="ovr",
    )
    summary_dict["matthews_correlation_coefficient"] = matthews_corrcoef(
        np.argmax(true_logits, axis=-1),
        np.argmax(pred_softmax, axis=-1),
    )

    return summary_dict
