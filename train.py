from model import customModelWithLocalization
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

NAME_BACKBONE = "yolo_v8_xs_backbone"
TRAIN = True

images = np.load("matrices_training.npy")

labels = np.load("labels_training.npy")

images_validation = np.load("matrices_validation.npy")

labels_validation = np.load("labels_validation.npy")

images_test = np.load("matrices_test.npy")

labels_test = np.load("labels_test.npy")

model = customModelWithLocalization(30)

def sort_labels(labels):
    labels_sorted = []
    for label in labels:
        array_tmp = []
        for elements in label:
            array_tmp.append(elements)
        # TO DO sort array
    return 0

def coordinates_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true[:, :, :2], y_pred[:, :, :2])

def confidence_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true[:, :, 2:], y_pred[:, :, 2:])

# Compile the model with separate loss functions for each output
model.compile(optimizer='adam', loss=[coordinates_loss, confidence_loss], metrics=['accuracy'])

if os.path.isfile(NAME_BACKBONE+".h5") and not TRAIN:
    model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)

    # Get predictions using the model
    print(model.predict(images_test[:1]))
    print(len(labels_test))
    print("Classes : ", labels_test[0][:1])

else:
    # Train model
    model.compile(optimizer='adam', loss=[coordinates_loss, confidence_loss], metrics=['accuracy'])

    label_test = sort_labels(labels)

    model.fit(images, label_test, validation_data=(images_validation, labels_validation), epochs=50, batch_size=32)

    model.save_weights(NAME_BACKBONE+".h5", overwrite="True", save_format="h5", options=None)


# model.summary()

# tf.keras.saving.save_model(model, "model.tf")
