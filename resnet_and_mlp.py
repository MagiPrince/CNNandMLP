import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, AveragePooling2D, BatchNormalization, Activation, Add
from keras.models import Model

# Resuidal block BN -> relu -> conv -> bn -> relu -> conv
def res_block(x, filters):
    bn1 = BatchNormalization()(x)
    act1 = Activation('relu')(bn1)
    conv1 = Conv2D(filters=filters, kernel_size=(3, 3), data_format='channels_first', strides=(2, 2), padding='same', 
                   kernel_initializer=glorot_uniform(seed=0))(act1)
    # print('conv1.shape', conv1.shape)
    bn2 = BatchNormalization()(conv1)
    act2 = Activation('relu')(bn2)
    conv2 = Conv2D(filters=filters, kernel_size=(3, 3), data_format='channels_first', strides=(1, 1), padding='same', 
                   kernel_initializer=glorot_uniform(seed=0))(act2)
    # print('conv2.shape', conv2.shape)
    residual = Conv2D(1, (1, 1), strides=(1, 1), data_format='channels_first')(conv2)
    
    
    x = Conv2D(filters=filters, kernel_size=(3, 3), data_format='channels_first', strides=(2, 2), padding='same', 
                   kernel_initializer=glorot_uniform(seed=0))(x)
    # print('x.shape', x.shape)
    out = Add()([x, residual])
    
    return out

def resnetModelWithLocalization(num_objects):
    # Input layer
    input1 = Input(shape=(64, 64, 3))

    # Combining resuidal blocks into a network
    res1 = res_block(input1, 64)
    # print('---------block 1 end-----------')
    res2 = res_block(res1, 128)
    # print('---------block 2 end-----------')
    res3 = res_block(res2, 256)
    # print('---------block 3 end-----------')
    res4 = res_block(res3, 512)
    # print('---------block 4 end-----------')

    # Classifier block
    act1 = Activation('relu')(res4)
    flatten1 = Flatten()(act1)
    dense1 = Dense(512)(flatten1)
    act2 = Activation('relu')(dense1)
    # dense2 = Dense(62, activation='relu')(act2)

    # MLP layers for object localization
    # x_dense = Dense(32, activation='relu')(dense2)
    outputs = []
    for _ in range(num_objects):
        output = Dense(2, activation='linear')(act2)  # Output x, y, confidence score
        # output = concatenate([output, Dense(1, activation='sigmoid')(act2)], axis=1)
        outputs.append(output)

    # Concatenate outputs and reshape to [batch_size, num_objects, 3]
    # outputs = [output for output in outputs]
    concatenated_outputs = outputs[0]
    for output in outputs[1:]:
        concatenated_outputs = concatenate([concatenated_outputs, output], axis=1)
    # concatenated_outputs = concatenate(outputs, axis=1)
    reshaped_outputs = Reshape((num_objects, 2))(concatenated_outputs)

    # Create the model
    model = Model(inputs=input1, outputs=reshaped_outputs)

    return model
