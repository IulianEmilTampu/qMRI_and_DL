"""
Script that defines the detection models used by the run_model_training_routine.py in the context
of the qMRI project and tumor detection
"""

import numpy as np
from typing import Union
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    MaxPool2D,
    GlobalAveragePooling2D,
)


def SimpleDetectionModel(
    num_classes: int,
    input_shape: Union[list, tuple],
    class_weights: Union[list, tuple] = None,
    kernel_size: Union[list, tuple] = (3, 3),
    pool_size: Union[list, tuple] = (2, 2),
    model_name: str = "SimpleDetectionModel",
    debug: bool = False,
):

    if class_weights is None:
        class_weights = np.ones([1, num_classes])
    else:
        class_weights = class_weights

    # building  model
    input = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=kernel_size, activation="relu", padding="same")(
        input
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=64, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=pool_size)(x)

    x = Conv2D(filters=128, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=128, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=pool_size)(x)

    x = Conv2D(filters=256, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=256, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=pool_size)(x)

    x = Conv2D(filters=512, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=512, kernel_size=kernel_size, activation="relu", padding="same")(
        x
    )
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(pool_size=pool_size)(x)

    # x = encoder(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(
        units=128,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    output = Dense(units=2, activation="softmax")(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    # print model if needed
    if debug is True:
        print(model.summary())

    return model
