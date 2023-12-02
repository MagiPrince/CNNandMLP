import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Concatenate, Reshape, AveragePooling2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D
from keras.models import Model
from qkeras import *
import contextlib

bits = 16
set_internal_sigmoid("real")

def basic_block(x, filters, strides=(1, 1)):
    shortcut = x
    
    # First convolution layer
    x = QConv2D(filters, (3, 3), strides=strides, padding='same',
                kernel_quantizer=quantized_bits(bits, 6, alpha=1),
                bias_quantizer=quantized_bits(bits, 6, alpha=1),
                use_bias=True)(x)
    x = QBatchNormalization()(x)
    x = QActivation(quantized_relu(bits, 6))(x)
    
    # Second convolution layer
    x = QConv2D(filters, (3, 3), strides=(1, 1), padding='same',
                kernel_quantizer=quantized_bits(bits, 6, alpha=1),
                bias_quantizer=quantized_bits(bits, 6, alpha=1),
                use_bias=True)(x)
    x = QBatchNormalization()(x)
    
    # Shortcut connection
    if strides != (1, 1) or shortcut.shape[-1] != filters:
        shortcut = QConv2D(filters, (1, 1), strides=strides, padding='same',
                kernel_quantizer=quantized_bits(bits, 6, alpha=1),
                bias_quantizer=quantized_bits(bits, 6, alpha=1),
                use_bias=True)(shortcut)
        shortcut = QBatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = QActivation(quantized_relu(bits, 6))(x)
    
    return x

def qresnetModelWithLocalization(num_objects):
    # Input layer
    input1 = Input(shape=(64, 64, 3))

    # Initial convolution layer
    x = QConv2D(64, (7, 7), strides=(2, 2), padding='same',
                kernel_quantizer=quantized_bits(bits, 6, alpha=1),
                bias_quantizer=quantized_bits(bits, 6, alpha=1),
                use_bias=True)(input1)
    x = QBatchNormalization()(x)
    x = QActivation(quantized_relu(bits, 6))(x)
    # x = QConv2D(2, (3, 3), strides=(3, 3), padding='same',
    #             kernel_quantizer=quantized_bits(bits, 6, alpha=1),
    #             bias_quantizer=quantized_bits(bits, 6, alpha=1),
    #             use_bias=True)(x)
    # x = QBatchNormalization()(x)
    # x = QActivation(quantized_relu(bits, 6))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
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

    x = GlobalAveragePooling2D()(x)

    x = Flatten()(x)

    output_coords = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
                        bias_quantizer=quantized_bits(bits, 0, alpha=1),
                        use_bias=True)(x) # Output : x, y
    output_conf = QDense(1, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
                        bias_quantizer=quantized_bits(bits, 0, alpha=1),
                        use_bias=True)(x) # Output : confidence
    concatenated_outputs = concatenate([QActivation(quantized_relu(bits, 6))(output_coords), QActivation(quantized_sigmoid(bits))(output_conf)], axis=1)
    for _ in range(num_objects-1):
        output_coords = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
                        bias_quantizer=quantized_bits(bits, 0, alpha=1),
                        use_bias=True)(x) # Output : x, y
        output_conf = QDense(1, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
                        bias_quantizer=quantized_bits(bits, 0, alpha=1),
                        use_bias=True)(x) # Output : confidence
        output = concatenate([QActivation(quantized_relu(bits, 6))(output_coords), QActivation(quantized_sigmoid(bits))(output_conf)], axis=1)
        concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output])


    reshaped_outputs = Reshape((num_objects, 3))(concatenated_outputs)

    # Create the model
    model = Model(inputs=input1, outputs=reshaped_outputs)

    return model
