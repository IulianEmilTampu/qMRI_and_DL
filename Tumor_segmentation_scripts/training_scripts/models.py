import os
import sys
import time
import json
import numpy as np
from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    BatchNormalization,
    MaxPooling2D,
    Conv2DTranspose,
    Dropout,
    LeakyReLU,
)
from tensorflow_addons.layers import InstanceNormalization

# custom imports
import utilities
from utilities import PrintModelSegmentation
from tensorflow_addons.optimizers import Lookahead
import tensorflow_addons as tfa


class unet(object):
    def __init__(
        self,
        img_size,
        Nclasses,
        class_weights,
        model_name="myWeightsAug.h5",
        Nfilter_start=64,
        depth=3,
        batch_size=3,
    ):
        self.img_size = img_size
        self.Nclasses = Nclasses
        self.class_weights = class_weights
        self.model_name = model_name
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        self.batch_size = batch_size

        self.model = Sequential()
        inputs = Input(img_size)

        def dice(y_true, y_pred, eps=1e-5):
            num = 2.0 * K.sum(
                self.class_weights * K.sum(y_true * y_pred, axis=[0, 1, 2])
            )
            den = (
                K.sum(self.class_weights * K.sum(y_true + y_pred, axis=[0, 1, 2])) + eps
            )
            return num / den

        def diceLoss(y_true, y_pred):
            return 1 - dice(y_true, y_pred)

        def bceLoss(y_true, y_pred):
            bce = K.sum(
                -self.class_weights * K.sum(y_true * K.log(y_pred), axis=[0, 1, 2])
            )
            return bce

        # This is a help function that performs 2 convolutions, each followed by batch normalization
        # and ReLu activations, Nf is the number of filters, filter size (3 x 3)
        def convs(layer, Nf):
            x = Conv2D(Nf, (3, 3), kernel_initializer="he_normal", padding="same")(
                layer
            )
            x = InstanceNormalization()(x)
            # x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.01)(x)
            x = Conv2D(Nf, (3, 3), kernel_initializer="he_normal", padding="same")(x)
            x = InstanceNormalization()(x)
            # x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.01)(x)
            return x

        # This is a help function that defines what happens in each layer of the encoder (downstream),
        # which calls "convs" and then Maxpooling (2 x 2). Save each layer for later concatenation in the upstream.
        def encoder_step(layer, Nf):
            y = convs(layer, Nf)
            # x = MaxPooling2D(pool_size=(2, 2))(y)
            x = Conv2D(
                Nf, (3, 3), strides=2, kernel_initializer="he_normal", padding="same"
            )(y)
            return y, x

        # This is a help function that defines what happens in each layer of the decoder (upstream),
        # which contains transpose convolution (filter size (3 x 3), stride (2,2) batch normalization, concatenation with
        # corresponding layer (y) from encoder, and lastly "convs"
        def decoder_step(layer, layer_to_concatenate, Nf):
            x = Conv2DTranspose(
                filters=Nf,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                kernel_initializer="he_normal",
            )(layer)
            x = InstanceNormalization()(x)
            x = tf.concat([x, layer_to_concatenate], axis=-1)
            x = convs(x, Nf)
            return x

        layers_to_concatenate = []
        x = inputs

        # Make encoder with 'self.depth' layers,
        # note that the number of filters in each layer will double compared to the previous "step" in the encoder
        for d in range(self.depth - 1):
            y, x = encoder_step(x, self.Nfilter_start * np.power(2, d))
            layers_to_concatenate.append(y)

        # Make bridge, that connects encoder and decoder using "convs" between them.
        # Use Dropout before and after the bridge, for regularization. Use dropout probability of 0.2.
        x = Dropout(0.3)(x)
        x = convs(x, self.Nfilter_start * np.power(2, self.depth - 1))
        x = Dropout(0.3)(x)

        # Make decoder with 'self.depth' layers,
        # note that the number of filters in each layer will be halved compared to the previous "step" in the decoder
        for d in range(self.depth - 2, -1, -1):
            y = layers_to_concatenate.pop()
            x = decoder_step(x, y, self.Nfilter_start * np.power(2, d))

        # Make classification (segmentation) of each pixel, using convolution with 1 x 1 filter
        final = Conv2D(filters=self.Nclasses, kernel_size=(1, 1), activation="softmax")(
            x
        )
        # Create model
        self.model = Model(inputs=inputs, outputs=final)
        self.model.compile(
            loss=diceLoss,
            optimizer=Adam(learning_rate=1e-4),
            metrics=["accuracy", dice],
        )

    def evaluate(self, X, Y):
        print("Evaluation process:")
        score, acc, dice = self.model.evaluate(X, Y, self.batch_size)
        print("Accuracy: {:.4f}".format(acc * 100))
        print("Dice: {:.4f}".format(dice))
        return acc, dice

    def predict(self, X):
        print("Segmenting unseen image")
        segmentation = self.model.predict(X, self.batch_size)
        return segmentation

    ## CUSTOM TRAINING FUNCTION
    def custum_train(
        self,
        train_gen,
        val_gen,
        train_steps,
        val_steps,
        max_epocs=300,
        verbose=0,
        save_model_path=None,
        early_stopping=None,
        patience=None,
        start_learning_rate=0.0001,
    ):
        """
        Custom training loop.
        """
        self.save_model_path = save_model_path
        self.verbose = verbose
        self.maxEpochs = max_epocs
        self.start_learning_rate = start_learning_rate
        self.train_steps = int(np.ceil(train_steps))
        self.val_steps = int(np.ceil(val_steps))
        self.early_stopping = early_stopping
        self.patience = patience

        if verbose <= 2 and isinstance(verbose, int):
            self.verbose = verbose
        else:
            print(
                "Invalid verbose parameter. Given {} but expected 0, 1 or 2. Setting to default 1".format(
                    verbose
                )
            )

        if early_stopping:
            self.best_acc = 0.0
            self.best_dice = 0.0
            n_wait = 0

        def dice(y_true, y_pred, class_weights, eps=1e-5):
            num = 2.0 * K.sum(class_weights * K.sum(y_true * y_pred, axis=[0, 1, 2]))
            den = K.sum(class_weights * K.sum(y_true + y_pred, axis=[0, 1, 2])) + eps
            return num / den

        def diceLoss(y_true, y_pred, class_weights):
            return 1 - dice(y_true, y_pred, class_weights)

        def bceLoss(model, x, y, class_weights, training):
            y_ = model(x, training=training)
            bce = K.sum(K.binary_crossentropy(y, y_), axis=[0, 1, 2])
            # bce = K.sum(-class_weights*K.sum(y*K.log(y_), axis=[0,1,2]))
            return bce

        def segmentationLoss(model, x, y, class_weights, training):
            # training=training is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            y_ = model(x, training=training)
            return diceLoss(y_true=y, y_pred=y_, class_weights=class_weights)

        def grad(model, inputs, targets, class_weights):
            with tf.GradientTape() as tape:
                segmentation_loss = segmentationLoss(
                    model, inputs, targets, class_weights, training=True
                )
                bce_loss = bceLoss(model, inputs, targets, class_weights, training=True)
                loss_value = segmentation_loss + bce_loss
                # loss_value = segmentationLoss(model, inputs, targets, class_weights, training=True)
            return loss_value, tape.gradient(loss_value, model.trainable_variables)

        # Keep results for plotting
        self.train_loss_history = []
        self.train_accuracy_history = []

        self.val_loss_history = []
        self.val_accuracy_history = []

        self.train_f1_history = []
        self.val_f1_history = []

        self.train_dice_history = []
        self.val_dice_history = []

        # start = time.time()

        # initialize the variables
        tr_epoch_loss_avg = tf.keras.metrics.Mean()
        tr_epoch_accuracy = tf.keras.metrics.Accuracy()
        tr_epoch_dice = []
        val_epoch_loss_avg = tf.keras.metrics.Mean()
        val_epoch_accuracy = tf.keras.metrics.Accuracy()
        val_epoch_dice = []

        # start looping through the epochs
        start = time.time()
        for epoch in range(self.maxEpochs):
            # reset metrics (keep only values for one epoch at the time)
            tr_epoch_loss_avg.reset_states()
            tr_epoch_accuracy.reset_states()
            tr_epoch_dice = []
            # tr_epoch_f1.reset_states()
            val_epoch_loss_avg.reset_states()
            val_epoch_accuracy.reset_states()
            val_epoch_dice = []

            # update learning rate based on the epoch
            self.learning_rate = self.start_learning_rate * np.power(
                1 - epoch / self.maxEpochs, 0.9
            )

            # ######## USING LOOKAHEAD OPTIMIZER
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            optimizer = Lookahead(optimizer, sync_period=5, slow_step_size=0.5)

            # ####### TRAINING
            step = 0
            epoch_start = time.time()
            for x, y in train_gen:
                step += 1

                # save information about training image size
                if epoch == 0 and step == 1:
                    self.batch_size = x.shape[0]
                    self.input_size = (x.shape[1], x.shape[2])

                train_loss, grads = grad(self.model, x, y, self.class_weights)

                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                # Track loss and accuracy
                tr_epoch_loss_avg.update_state(train_loss)
                tr_epoch_accuracy.update_state(
                    y.reshape((-1, 1)),
                    self.model(x, training=False).numpy().reshape((-1, 1)),
                )
                tr_epoch_dice.append(
                    dice(y, self.model(x, training=False), self.class_weights).numpy()
                )

                # print values
                if self.verbose == 2:
                    print(
                        f"Epoch {epoch+1:04d} training "
                        f"{step:04d}/{self.train_steps:04d} -> "
                        f'{"tr_loss":8s}:{tr_epoch_loss_avg.result():.4f}, '
                        f'{"tr_acc":8s}:{tr_epoch_accuracy.result():.4f}, '
                        f'{"tr_dice":8s}:{np.mean(tr_epoch_dice):.4f}\r',
                        end="",
                    )
                if step >= self.train_steps:
                    break

            # finisced all the training batches -> save training loss
            self.train_loss_history.append(
                tr_epoch_loss_avg.result().numpy().astype(float)
            )
            self.train_accuracy_history.append(
                tr_epoch_accuracy.result().numpy().astype(float)
            )
            self.train_dice_history.append(np.mean(tr_epoch_dice).astype(float))

            # ########### VALIDATION
            step = 0
            for x, y in val_gen:
                step += 1

                val_loss = segmentationLoss(
                    self.model, x, y, self.class_weights, training=False
                ) + bceLoss(self.model, x, y, self.class_weights, training=False)

                # track progress
                val_epoch_loss_avg.update_state(val_loss)
                val_epoch_accuracy.update_state(
                    y.reshape((-1, 1)),
                    self.model(x, training=False).numpy().reshape((-1, 1)),
                )
                val_epoch_dice.append(
                    dice(y, self.model(x, training=False), self.class_weights).numpy()
                )

                # print values
                if self.verbose == 2:
                    print(
                        f"Epoch {epoch+1:04d} validation "
                        f"{step:04d}/{self.val_steps:04d} -> "
                        f'{"val_loss":8s}:{val_epoch_loss_avg.result():.4f}, '
                        f'{"val_acc":8s}:{val_epoch_accuracy.result():.4f}, '
                        f'{"val_dice":8s}:{np.mean(val_epoch_dice):.4f}\r',
                        end="",
                    )

                if step >= self.val_steps:
                    break

            # finisced all the batches in the validation
            self.val_loss_history.append(
                val_epoch_loss_avg.result().numpy().astype(float)
            )
            self.val_accuracy_history.append(
                val_epoch_accuracy.result().numpy().astype(float)
            )
            self.val_dice_history.append(np.mean(val_epoch_dice).astype(float))
            # self.val_f1_history.append(val_epoch_f1.result().numpy().astype(float))

            # print averall information for this epoch
            epoch_stop = time.time()
            if any([self.verbose == 1, self.verbose == 2]):
                # clear line
                sys.stdout.write("\033[K")
                print(
                    f"Epoch {epoch+1:04d} -> "
                    f'{"tr_loss":8s}:{self.train_loss_history[-1]:.4f}, '
                    f'{"tr_acc":8s}:{self.train_accuracy_history[-1]:.4f}, '
                    f'{"tr_dice":8s}:{self.train_dice_history[-1]:.4f}\n'
                    f'{" "*14}'
                    f'{"val_loss":8s}:{self.val_loss_history[-1]:.4f}, '
                    f'{"val_acc":8s}:{self.val_accuracy_history[-1]:.4f}, '
                    f'{"val_dice":8s}:{self.val_dice_history[-1]:.4f}\n',
                    f"learning rate:{self.learning_rate:0.5f}\n",
                    f"elapsed:{utilities.tictoc(epoch_start,epoch_stop)}\n\r",
                    end="",
                )

            # plot some information
            if epoch % 50 == 0:
                utilities.plotModelPerformance(
                    self.train_loss_history,
                    self.train_accuracy_history,
                    self.val_loss_history,
                    self.val_accuracy_history,
                    self.train_dice_history,
                    self.val_dice_history,
                    self.save_model_path,
                    best_epoch=None,
                    display=False,
                )
                try:
                    utilities.plotEpochSegmentation(
                        x,
                        y,
                        self.model(x, training=False).numpy(),
                        save_path=self.save_model_path,
                        epoch=epoch,
                        display=False,
                    )
                except:
                    print(f"Skipping printing for epoch {epoch}")

            if self.early_stopping:
                # check if model accurary improved, and update counter if needed
                if self.val_dice_history[-1] > self.best_dice:
                    # save model checkpoint
                    if any([self.verbose == 1, self.verbose == 2]):
                        print(f" - Saving model checkpoint in {self.save_model_path}")
                    # save some extra parameters
                    stop = time.time()
                    self.training_time = utilities.tictoc_from_time(stop - start)
                    self.training_epochs = epoch
                    self.best_acc = self.val_accuracy_history[-1]
                    # self.best_f1 = self.val_f1_history[-1]
                    self.best_epoch = epoch
                    self.best_dice = self.val_dice_history[-1]

                    # save model
                    self.model.save(os.path.join(self.save_model_path, "best_model.tf"))
                    self.model.save_weights(
                        os.path.join(self.save_model_path, "best_model_weights.tf")
                    )

                    # save history training curves
                    json_dict = OrderedDict()
                    json_dict["training_loss"] = list(self.train_loss_history)
                    json_dict["validation_loss"] = list(self.val_loss_history)
                    json_dict["training_accuracy"] = list(self.train_accuracy_history)
                    json_dict["validation_accuracy"] = list(self.val_accuracy_history)
                    json_dict["training_dice"] = list(self.train_dice_history)
                    json_dict["validation_dice"] = list(self.val_dice_history)
                    with open(os.path.join(save_model_path, "history.json"), "w") as fp:
                        json.dump(json_dict, fp)

                    # reset counter
                    n_wait = 0
                else:
                    n_wait += 1
                # check max waiting is reached
                if n_wait == self.patience:
                    if any([self.verbose == 1, self.verbose == 2]):
                        print(
                            " -  Early stopping patient reached. Best model saved in {}".format(
                                self.save_model_path
                            )
                        )
                    # save history training curves
                    json_dict = OrderedDict()
                    json_dict["training_loss"] = list(self.train_loss_history)
                    json_dict["validation_loss"] = list(self.val_loss_history)
                    json_dict["training_accuracy"] = list(self.train_accuracy_history)
                    json_dict["validation_accuracy"] = list(self.val_accuracy_history)
                    json_dict["training_dice"] = list(self.train_dice_history)
                    json_dict["validation_dice"] = list(self.val_dice_history)
                    with open(os.path.join(save_model_path, "history.json"), "w") as fp:
                        json.dump(json_dict, fp)
                    # save training curves
                    utilities.plotModelPerformance(
                        self.train_loss_history,
                        self.train_accuracy_history,
                        self.val_loss_history,
                        self.val_accuracy_history,
                        self.train_dice_history,
                        self.val_dice_history,
                        self.save_model_path,
                        best_epoch=None,
                        display=False,
                    )

                    break

            # save last model as well
            if epoch == self.maxEpochs - 1:
                if any([self.verbose == 1, self.verbose == 2]):
                    print(
                        " -  Run through all the epochs. Last model saved in {}".format(
                            self.save_model_path
                        )
                    )
                # save model
                self.model.save(os.path.join(self.save_model_path, "last_model.tf"))
                self.model.save_weights(
                    os.path.join(self.save_model_path, "last_model_weights.tf")
                )
                # save history training curves
                json_dict = OrderedDict()
                json_dict["training_loss"] = list(self.train_loss_history)
                json_dict["validation_loss"] = list(self.val_loss_history)
                json_dict["training_accuracy"] = list(self.train_accuracy_history)
                json_dict["validation_accuracy"] = list(self.val_accuracy_history)
                json_dict["training_dice"] = list(self.train_dice_history)
                json_dict["validation_dice"] = list(self.val_dice_history)
                with open(os.path.join(save_model_path, "history.json"), "w") as fp:
                    json.dump(json_dict, fp)
                # save training curves
                utilities.plotModelPerformance(
                    self.train_loss_history,
                    self.train_accuracy_history,
                    self.val_loss_history,
                    self.val_accuracy_history,
                    self.train_dice_history,
                    self.val_dice_history,
                    self.save_model_path,
                    best_epoch=None,
                    display=False,
                )
                break
