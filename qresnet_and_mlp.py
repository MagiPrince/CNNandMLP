import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, AveragePooling2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D
from keras.models import Model
from qkeras import *
import contextlib

bits = 8

@contextlib.contextmanager
def options(options):
  old_opts = tf.config.optimizer.get_experimental_options()
  tf.config.optimizer.set_experimental_options(options)
  try:
    yield
  finally:
    tf.config.optimizer.set_experimental_options(old_opts)

def basic_block(x, filters, strides=(1, 1)):
    # shortcut = x
    
    # First convolution layer
    x = QConv2D(filters, (3, 3), strides=strides, padding='same',
                kernel_quantizer=quantized_bits(bits, 0, alpha=1),
                bias_quantizer=quantized_bits(bits, 0, alpha=1),
                kernel_initializer='lecun_uniform', use_bias=True)(x)
    x = QBatchNormalization()(x)
    x = QActivation(quantized_relu(bits))(x)
    
    # Second convolution layer
    x = QConv2D(filters, (3, 3), strides=(1, 1), padding='same',
                kernel_quantizer=quantized_bits(bits, 0, alpha=1),
                bias_quantizer=quantized_bits(bits, 0, alpha=1),
                kernel_initializer='lecun_uniform', use_bias=True)(x)
    x = QBatchNormalization()(x)
    
    # Shortcut connection
    # if strides != (1, 1) or shortcut.shape[-1] != filters:
    #     shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
    #     shortcut = BatchNormalization()(shortcut)
    
    # x = Add()([x, shortcut])
    x = QActivation(quantized_relu(bits))(x)
    
    return x

def qresnetModelWithLocalization(num_objects):
    # with options({"layout_optimizer": False}):
    # Input layer
    input1 = Input(shape=(64, 64, 3))

    # Initial convolution layer
    x = QConv2D(2, (7, 7), strides=(2, 2), padding='same',
                kernel_quantizer=quantized_bits(bits, 0, alpha=1),
                bias_quantizer=quantized_bits(bits, 0, alpha=1),
                kernel_initializer='lecun_uniform', use_bias=True)(input1)
    x = QBatchNormalization()(x)
    x = QActivation(quantized_relu(bits))(x)
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
    
    x = Flatten()(x)
    
    # outputs = []
    # for _ in range(num_objects):
    #     output = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
    #                     bias_quantizer=quantized_bits(bits, 0, alpha=1),
    #                     kernel_initializer='lecun_uniform', use_bias=True)(x)  # Output : x, y, w, h
    #     # output = Activation('relu')(output)
    #     output = QActivation(quantized_relu(bits, 6))(output)
    #     # output = concatenate([output, Dense(1, activation='sigmoid')(x)], axis=1) # Output : x, y, w, h, confidence score
    #     outputs.append(output)
    #     x = Flatten()(x)

    # # Concatenate outputs and reshape to [batch_size, num_objects, 5]
    # concatenated_outputs = outputs[0]
    # for output in outputs[1:]:
    #     concatenated_outputs = concatenate([concatenated_outputs, output], axis=1)
    # reshaped_outputs = Reshape((num_objects, 2))(concatenated_outputs)

    # output = Dense(2, activation='relu')(x)

    output = QDense(2, kernel_quantizer= quantized_bits(bits, 0, alpha=1),
                        bias_quantizer=quantized_bits(bits, 0, alpha=1),
                        kernel_initializer='lecun_uniform', use_bias=True)(x)
    output = QActivation(quantized_relu(bits, 6))(output)

    # Create the model
    model = Model(inputs=input1, outputs=output)

    return model

