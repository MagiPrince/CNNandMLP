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
    x_dense = Dense(128, activation='relu')(x)
    x_dense = Dense(64, activation='relu')(x_dense)
    x_dense = Dense(32, activation='relu')(x_dense)
    outputs = []
    for _ in range(num_objects):
        output = Dense(2, activation='linear')(x_dense)  # Output x, y, confidence score
        output = concatenate([output, Dense(1, activation='sigmoid')(x_dense)], axis=1)
        outputs.append(output)

    # Concatenate outputs and reshape to [batch_size, num_objects, 3]
    # outputs = [output for output in outputs]
    concatenated_outputs = outputs[0]
    for output in outputs[1:]:
        concatenated_outputs = concatenate([concatenated_outputs, output], axis=1)
    # concatenated_outputs = concatenate(outputs, axis=1)
    reshaped_outputs = Reshape((num_objects, 3))(concatenated_outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=reshaped_outputs)

    return model

# # Create the custom model with object localization
# num_objects = 10

# model = customModelWithLocalization(num_objects)
# # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# # Define individual loss functions for coordinates and confidence scores
# def coordinates_loss(y_true, y_pred):
#     return tf.keras.losses.mean_squared_error(y_true[:, :, :2], y_pred[:, :, :2])

# def confidence_loss(y_true, y_pred):
#     return tf.keras.losses.binary_crossentropy(y_true[:, :, 2:], y_pred[:, :, 2:])

# # Compile the model with separate loss functions for each output
# model.compile(optimizer='adam', loss=[coordinates_loss, confidence_loss], metrics=['accuracy'])

# # Generate example input data
# num_samples = 2  # Small number of samples for illustration
# X_train = np.random.rand(num_samples, 64, 64, 3)

# # Generate example labels (for illustration)
# y_train = np.random.rand(num_samples, num_objects, 3)

# print(y_train)

# # Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=2, verbose=1)

# # Example inference
# output_ = model.predict(X_train)
# print("Model output (x, y, confidence) for each object:")
# print(output_)
