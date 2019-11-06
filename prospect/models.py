from keras import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam, SGD
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# simple deep architecture for numeric/categorical data




def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(16, input_dim=dim, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(4, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="sigmoid"))

    # return our model
    return model


def create_cnn(width, height, depth,  filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)
    kernel_shapes = [3, 3, 3]
    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (kernel_shapes[i], kernel_shapes[i]), padding="same")(x)
        x = Activation("relu")(x)
        # x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    # x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="sigmoid")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model


def create_mixed(model_list, direct=False):
    if direct:
        inputs = Input(shape=[2,])
        combinedInput = concatenate([inputs] + [model.output for model in model_list])
    else:
        combinedInput = concatenate([model.output for model in model_list])


    # # our final FC layer head will have two dense layers, the final one
    # # being our classification head
    # x = BatchNormalization(axis=-1)(combinedInput)
    x = Dense(32, activation="relu")(combinedInput)
    x = Dense(16, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)

    # our final model will accept numerical data on the MLP
    # input and images on the CNN input, outputting a single value
    if direct:
        model = Model(inputs=[inputs] + [model.input for model in model_list], outputs=x)
    else:
        model = Model(inputs=[model.input for model in model_list], outputs=x)

    # compile the model using binary crossentropy  as our loss,

    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


