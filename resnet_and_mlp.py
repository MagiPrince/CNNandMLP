import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, AveragePooling2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D
from keras.models import Model

def basic_block(x, filters, strides=(1, 1)):
    # shortcut = x
    
    # First convolution layer
    x = Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second convolution layer
    x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Shortcut connection
    # if strides != (1, 1) or shortcut.shape[-1] != filters:
    #     shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
    #     shortcut = BatchNormalization()(shortcut)
    
    # x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def resnetModelWithLocalization(num_objects):
    # Input layer
    input1 = Input(shape=(64, 64, 3))

    # Initial convolution layer
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3))(x)
    # x = Conv2D(1, (3, 3), strides=(2, 2), padding='same')(x)
    
    # Residual blocks
    x = basic_block(x, 64)
    x = basic_block(x, 64)
    x = basic_block(x, 128, strides=(2, 2))
    x = basic_block(x, 128)
    x = basic_block(x, 256, strides=(2, 2))
    x = basic_block(x, 256)
    x = basic_block(x, 512, strides=(2, 2))
    x = basic_block(x, 512)

    print(x.shape)

    x = Conv2D(512, (2, 2), padding='same')(x)

    print(x.shape)
    
    x = Flatten()(x)
    
    # outputs = []
    concatenated_outputs = Dense(2, activation='relu')(x)
    for _ in range(num_objects-1):
        output = Dense(2, activation='relu')(x)  # Output : x, y
        concatenated_outputs = concatenate([concatenated_outputs, output], axis=1)

    reshaped_outputs = Reshape((num_objects, 2))(concatenated_outputs)

    # Create the model
    model = Model(inputs=input1, outputs=reshaped_outputs)

    return model
