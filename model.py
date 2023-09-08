import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, AveragePooling2D
from keras.models import Model

def customModelWithLocalization(num_objects):
    # Input layer
    inputs = Input(shape=(64, 64, 3))

    # CNN layers
    x = Conv2D(16, (3, 3), activation='relu')(inputs)
    x = AveragePooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # MLP layers for object localization
    outputs = []
    for _ in range(num_objects):
        output = Dense(4, activation='relu')(x)  # Output x, y, confidence score
        output = concatenate([output, Dense(1, activation='sigmoid')(x)], axis=1)
        outputs.append(output)

    # Concatenate outputs and reshape to [batch_size, num_objects, 3]
    # outputs = [output for output in outputs]
    concatenated_outputs = outputs[0]
    for output in outputs[1:]:
        concatenated_outputs = concatenate([concatenated_outputs, output], axis=1)
    # concatenated_outputs = concatenate(outputs, axis=1)
    reshaped_outputs = Reshape((num_objects, 5))(concatenated_outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=reshaped_outputs)

    return model

