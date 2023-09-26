import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Concatenate, Reshape, AveragePooling2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D
from keras.models import Model
from qkeras import *
import contextlib

bits = 16

def basic_block(x, filters, strides=(1, 1)):
    # shortcut = x
    
    # First convolution layer
    x = QConv2D(filters, (3, 3), strides=strides, padding='same',
                kernel_quantizer=quantized_bits(bits, 6, alpha=1),
                bias_quantizer=quantized_bits(bits, 6, alpha=1),
                kernel_initializer='lecun_uniform', use_bias=True)(x)
    x = QBatchNormalization()(x)
    x = QActivation(quantized_relu(bits, 6))(x)
    
    # Second convolution layer
    x = QConv2D(filters, (3, 3), strides=(1, 1), padding='same',
                kernel_quantizer=quantized_bits(bits, 6, alpha=1),
                bias_quantizer=quantized_bits(bits, 6, alpha=1),
                kernel_initializer='lecun_uniform', use_bias=True)(x)
    x = QBatchNormalization()(x)
    
    # Shortcut connection
    # if strides != (1, 1) or shortcut.shape[-1] != filters:
    #     shortcut = QConv2D(filters, (1, 1), strides=strides, padding='same',
    #             kernel_quantizer=quantized_bits(bits, 6, alpha=1),
    #             bias_quantizer=quantized_bits(bits, 6, alpha=1),
    #             kernel_initializer='lecun_uniform', use_bias=True)(shortcut)
    #     shortcut = QBatchNormalization()(shortcut)
    
    # x = Add()([x, shortcut])
    x = QActivation(quantized_relu(bits, 6))(x)
    
    return x

def qresnetModelWithLocalization(num_objects):
    # Input layer
    input1 = Input(shape=(64, 64, 3))

    # Initial convolution layer
    x = QConv2D(2, (7, 7), strides=(2, 2), padding='same',
                kernel_quantizer=quantized_bits(bits, 6, alpha=1),
                bias_quantizer=quantized_bits(bits, 6, alpha=1),
                kernel_initializer='lecun_uniform', use_bias=True)(input1)
    x = QBatchNormalization()(x)
    x = QActivation(quantized_relu(bits, 6))(x)
    x = QConv2D(2, (3, 3), strides=(3, 3), padding='same',
                kernel_quantizer=quantized_bits(bits, 6, alpha=1),
                bias_quantizer=quantized_bits(bits, 6, alpha=1),
                kernel_initializer='lecun_uniform', use_bias=True)(x)
    x = QBatchNormalization()(x)
    x = QActivation(quantized_relu(bits, 6))(x)
    # x = MaxPooling2D((3, 3))(x)
    # x = Conv2D(1, (3, 3), strides=(2, 2), padding='same')(x)
    
    # Residual blocks
    x = basic_block(x, 4)
    x = basic_block(x, 4)
    x = basic_block(x, 8, strides=(2, 2))
    x = basic_block(x, 8)
    x = basic_block(x, 16, strides=(2, 2))
    x = basic_block(x, 16)
    x = basic_block(x, 32, strides=(2, 2))
    x = basic_block(x, 32)

    # x_2 = QConv2D(32, (1, 1), strides=(1, 1), padding='same',
    #             kernel_quantizer=quantized_bits(bits, 6, alpha=1),
    #             bias_quantizer=quantized_bits(bits, 6, alpha=1),
    #             kernel_initializer='lecun_uniform', use_bias=True)(x)
    # x_2 = QBatchNormalization()(x_2)
    # x_2 = QActivation(quantized_relu(bits, 6))(x_2)

    # x_3 = QConv2D(32, (1, 1), strides=(1, 1), padding='same',
    #             kernel_quantizer=quantized_bits(bits, 6, alpha=1),
    #             bias_quantizer=quantized_bits(bits, 6, alpha=1),
    #             kernel_initializer='lecun_uniform', use_bias=True)(x_2)
    # x_3 = QBatchNormalization()(x_3)
    # x_3 = QActivation(quantized_relu(bits, 6))(x_3)

    # x_4 = QConv2D(32, (1, 1), strides=(1, 1), padding='same',
    #             kernel_quantizer=quantized_bits(bits, 6, alpha=1),
    #             bias_quantizer=quantized_bits(bits, 6, alpha=1),
    #             kernel_initializer='lecun_uniform', use_bias=True)(x_3)
    # x_4 = QBatchNormalization()(x_4)
    # x_4 = QActivation(quantized_relu(bits, 6))(x_4)
    
    # x_5 = QConv2D(32, (1, 1), strides=(1, 1), padding='same',
    #             kernel_quantizer=quantized_bits(bits, 6, alpha=1),
    #             bias_quantizer=quantized_bits(bits, 6, alpha=1),
    #             kernel_initializer='lecun_uniform', use_bias=True)(x_4)
    # x_5 = QBatchNormalization()(x_5)
    # x_5 = QActivation(quantized_relu(bits, 6))(x_5)

    # x_6 = QConv2D(32, (1, 1), strides=(1, 1), padding='same',
    #             kernel_quantizer=quantized_bits(bits, 6, alpha=1),
    #             bias_quantizer=quantized_bits(bits, 6, alpha=1),
    #             kernel_initializer='lecun_uniform', use_bias=True)(x_5)
    # x_6 = QBatchNormalization()(x_6)
    # x_6 = QActivation(quantized_relu(bits, 6))(x_6)

    # x_7 = QConv2D(32, (1, 1), strides=(1, 1), padding='same',
    #             kernel_quantizer=quantized_bits(bits, 6, alpha=1),
    #             bias_quantizer=quantized_bits(bits, 6, alpha=1),
    #             kernel_initializer='lecun_uniform', use_bias=True)(x_6)
    # x_7 = QBatchNormalization()(x_7)
    # x_7 = QActivation(quantized_relu(bits, 6))(x_7)

    x = Flatten()(x)

    # x_1_flatten = Flatten()(x)

    # x_2_flatten = Flatten()(x_2)

    # x_3_flatten = Flatten()(x_3)

    # x_4_flatten = Flatten()(x_4)

    # x_5_flatten = Flatten()(x_5)

    # x_6_flatten = Flatten()(x_6)
    
    # x_7_flatten = Flatten()(x_7)

    output_1 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
                        bias_quantizer=quantized_bits(bits, 0, alpha=1),
                        kernel_initializer='lecun_uniform', use_bias=True)(x) # Output : x, y
    output_1 = QActivation(quantized_relu(bits, 6))(output_1)
    
    output_2 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
                        bias_quantizer=quantized_bits(bits, 0, alpha=1),
                        kernel_initializer='lecun_uniform', use_bias=True)(x) # Output : x, y
    output_2 = QActivation(quantized_relu(bits, 6))(output_2)

    concatenated_outputs = Concatenate(axis=-1)([output_1, output_2])

    

    output_3 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
                        bias_quantizer=quantized_bits(bits, 0, alpha=1),
                        kernel_initializer='lecun_uniform', use_bias=True)(x) # Output : x, y
    output_3 = QActivation(quantized_relu(bits, 6))(output_3)

    concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_3])

    # output_4 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_2_flatten) # Output : x, y
    # output_4 = QActivation(quantized_relu(bits, 6))(output_4)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_4])

    # # x_2 = QBatchNormalization()(x_2)
    # # x_3 = QActivation(quantized_relu(bits, 6))(x_2)

    # output_5 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_3_flatten) # Output : x, y
    # output_5 = QActivation(quantized_relu(bits, 6))(output_5)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_5])

    # output_6 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_3_flatten) # Output : x, y
    # output_6 = QActivation(quantized_relu(bits, 6))(output_6)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_6])

    # # x_3 = QBatchNormalization()(x_3)
    # # x_4 = QActivation(quantized_relu(bits, 6))(x_3)

    # output_7 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_4_flatten) # Output : x, y
    # output_7 = QActivation(quantized_relu(bits, 6))(output_7)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_7])

    # output_8 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_4_flatten) # Output : x, y
    # output_8 = QActivation(quantized_relu(bits, 6))(output_8)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_8])

    # output_9 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_5_flatten) # Output : x, y
    # output_9 = QActivation(quantized_relu(bits, 6))(output_9)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_9])

    # output_10 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_5_flatten) # Output : x, y
    # output_10 = QActivation(quantized_relu(bits, 6))(output_10)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_10])

    # output_11 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_6_flatten) # Output : x, y
    # output_11 = QActivation(quantized_relu(bits, 6))(output_11)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_11])

    # output_12 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_6_flatten) # Output : x, y
    # output_12 = QActivation(quantized_relu(bits, 6))(output_12)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_12])

    # output_13 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_7_flatten) # Output : x, y
    # output_13 = QActivation(quantized_relu(bits, 6))(output_13)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_13])

    # output_14 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x_7_flatten) # Output : x, y
    # output_14 = QActivation(quantized_relu(bits, 6))(output_14)

    # concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output_14])

    # output_1 = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x) # Output : x, y
    # concatenated_outputs = QActivation(quantized_relu(bits, 6))(output_1)
    # for _ in range(num_objects-1):
    #     output_tmp = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x) # Output : x, y
    #     output = QActivation(quantized_relu(bits, 6))(output_tmp)
    #     concatenated_outputs = Concatenate(axis=-1)([concatenated_outputs, output])


    reshaped_outputs = Reshape((3, 2))(concatenated_outputs)

    # Create the model
    model = Model(inputs=input1, outputs=reshaped_outputs)

    return model

