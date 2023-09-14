import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, AveragePooling2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D
from keras.models import Model
from qkeras import *

def basic_block(x, filters, strides=(1, 1)):
    # shortcut = x
    
    # First convolution layer
    x = QConv2D(filters, (3, 3), strides=strides, padding='same', kernel_quantizer=quantized_bits(6))(x)
    # x = QBatchNormalization()(x)
    x = QActivation('quantized_relu(6)')(x)
    
    # Second convolution layer
    x = QConv2D(filters, (3, 3), strides=(1, 1), padding='same', kernel_quantizer=quantized_bits(6))(x)
    # x = QBatchNormalization()(x)
    
    # Shortcut connection
    # if strides != (1, 1) or shortcut.shape[-1] != filters:
    #     shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
    #     shortcut = BatchNormalization()(shortcut)
    
    # x = Add()([x, shortcut])
    x = QActivation('quantized_relu(6)')(x)
    
    return x

def qresnetModelWithLocalization(num_objects):
    # Input layer
    input1 = Input(shape=(64, 64, 3))

    # Initial convolution layer
    x = QConv2D(1, (7, 7), strides=(2, 2), padding='same', kernel_quantizer=quantized_bits(6), activation='quantized_relu(6)')(input1)
    # x = QBatchNormalization()(x)
    # x = QActivation('quantized_relu(6)')(x)
    # x = QAveragePooling2D((3, 3), strides=(2, 2))(x)
    
    # Residual blocks
    x = basic_block(x, 1)
    x = basic_block(x, 1)
    x = basic_block(x, 2, strides=(2, 2))
    x = basic_block(x, 2)
    x = basic_block(x, 4, strides=(2, 2))
    x = basic_block(x, 4)
    x = basic_block(x, 8, strides=(2, 2))
    x = basic_block(x, 8)

    # x_2 = QConv2D(1, (3, 3), strides=(1, 1), padding='same', kernel_quantizer=quantized_bits(6), activation='quantized_relu(6)')(x_1)
    # # x = QActivation('quantized_relu(6)')(x)
    # x_3 = QConv2D(1, (3, 3), strides=(1, 1), padding='same', kernel_quantizer=quantized_bits(6), activation='quantized_relu(6)')(x_2)
    # # x = QActivation('quantized_relu(6)')(x)
    # x_4 = QConv2D(2, (3, 3), strides=(2, 2), padding='same', kernel_quantizer=quantized_bits(6), activation='quantized_relu(6)')(x_3)
    # # x = QActivation('quantized_relu(6)')(x)
    # x_5 = QConv2D(2, (3, 3), strides=(1, 1), padding='same', kernel_quantizer=quantized_bits(6), activation='quantized_relu(6)')(x_4)
    # # x = QActivation('quantized_relu(6)')(x)

    # x = MaxPooling2D()(x)
    
    x = Flatten()(x)

    # x = Dense(32, activation='quantized_relu(6)')(x)
    
    # outputs = []
    # for _ in range(num_objects):
    #     output = QDense(2, activation='quantized_relu(6)', kernel_quantizer=quantized_bits(6))(x) # Output : x, y
    #     outputs.append(output)

    # # Concatenate outputs and reshape to [batch_size, num_objects, 5]
    # concatenated_outputs = outputs[0]
    # for output in outputs[1:]:
    #     concatenated_outputs = concatenate([concatenated_outputs, output], axis=1)
    # reshaped_outputs = Reshape((num_objects, 2))(concatenated_outputs)

    output = QDense(2, activation='quantized_relu(6)', kernel_quantizer=quantized_bits(6))(x) # Output : x, y

    # output = QDense(2, activation='relu', kernel_quantizer=quantized_bits(3),
    #     bias_quantizer=quantized_bits(3))(x)

    # Create the model
    model = Model(inputs=input1, outputs=output)

    return model
