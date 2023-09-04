from model import customModelWithLocalization
from resnet_and_mlp import resnetModelWithLocalization
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from matplotlib import pyplot as plt

NAME_BACKBONE = "cnn_and_mlp"
TRAIN = True

images = np.load("matrices_training.npy")

labels = np.load("labels_training.npy")

model = resnetModelWithLocalization(30)

def coordinates_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true[:, :, :2], y_pred[:, :, :2])

def confidence_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true[:, :, 2:], y_pred[:, :, 2:])

# Compile the model with separate loss functions for each output
model.compile(optimizer='adam', loss=[coordinates_loss, confidence_loss], metrics=['accuracy'])

if os.path.isfile(NAME_BACKBONE+".h5") and not TRAIN:

    images_test = np.load("matrices_test.npy")

    labels_test = np.load("labels_test.npy")

    model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)

    # Get predictions using the model
    pred = model.predict(images_test[3:4])[0]
    print(pred.shape)
    print(pred[pred[:, 0].argsort()])
    lab = labels_test[3:4][0]
    print(lab[lab[:, 0].argsort()])

    model.summary()

    plt.imshow(images_test[3:4][0])
    plt.show()

else:
    # Train model
    model.compile(optimizer='adam', loss=[coordinates_loss, confidence_loss], metrics=['accuracy'])

    len_labels = len(labels)
    # sorted_labels = labels[:, labels[:, :, 0].argsort()][np.diag_indices(len_labels)]
    # sorted_labels = sorted_labels[:,:,:-1]
    # print(sorted_labels)

    images_validation = np.load("matrices_validation.npy")

    labels_validation = np.load("labels_validation.npy")
    labels_validation = labels_validation[:,:,:-1]

    model.fit(images, labels, validation_data=(images_validation, labels_validation), epochs=100, batch_size=32)

    model.save_weights(NAME_BACKBONE+".h5", overwrite="True", save_format="h5", options=None)


# model.summary()

# tf.keras.saving.save_model(model, "model.tf")
