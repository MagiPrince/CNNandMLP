import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, AveragePooling2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D
from keras.models import Model
from qkeras import *

def basic_block(x, filters, strides=(1, 1)):
    # shortcut = x
    
    # First convolution layer
    x = QConv2D(filters, (3, 3), strides=strides, padding='same')(x)
    x = QBatchNormalization()(x)
    x = QActivation('quantized_relu(16)')(x)
    
    # Second convolution layer
    x = QConv2D(filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = QBatchNormalization()(x)
    
    # Shortcut connection
    # if strides != (1, 1) or shortcut.shape[-1] != filters:
    #     shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
    #     shortcut = BatchNormalization()(shortcut)
    
    # x = Add()([x, shortcut])
    x = QActivation('quantized_relu(16)')(x)
    
    return x

def qresnetModelWithLocalization(num_objects):
    # Input layer
    input1 = Input(shape=(64, 64, 3))

    # Initial convolution layer
    x = QConv2D(64, (7, 7), strides=(2, 2), padding='same')(input1)
    x = QBatchNormalization()(x)
    x = QActivation('quantized_relu(16)')(x)
    # x = QAveragePooling2D((3, 3), strides=(2, 2))(x)
    
    # Residual blocks
    x = basic_block(x, 64)
    x = basic_block(x, 64)
    x = basic_block(x, 128, strides=(2, 2))
    x = basic_block(x, 128)
    x = basic_block(x, 256, strides=(2, 2))
    x = basic_block(x, 256)
    x = basic_block(x, 512, strides=(2, 2))
    x = basic_block(x, 512)
    
    x = Flatten()(x)

    outputs = []
    for _ in range(num_objects):
        output = QDense(2, activation='quantized_relu(16)')(x) # Output : x, y
        outputs.append(output)

    # Concatenate outputs and reshape to [batch_size, num_objects, 5]
    concatenated_outputs = outputs[0]
    for output in outputs[1:]:
        concatenated_outputs = concatenate([concatenated_outputs, output], axis=1)
    reshaped_outputs = Reshape((num_objects, 2))(concatenated_outputs)

    # Create the model
    model = Model(inputs=input1, outputs=reshaped_outputs)

    return model
