from model import customModelWithLocalization
from resnet_and_mlp import resnetModelWithLocalization
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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

    # model.summary()

    plt.imshow(images_test[3:4][0])
    plt.show()

else:
    # Train model
    model.compile(optimizer='adam', loss=[coordinates_loss], metrics=['accuracy'])

    len_labels = len(labels)
    tmp_labels = labels[:len(labels)//2]
    sorted_labels = tmp_labels[:, tmp_labels[:, :, 0].argsort()][np.diag_indices(len(tmp_labels))]
    tmp_labels = labels[len(labels)//2:]
    sorted_labels = sorted_labels + tmp_labels[:, tmp_labels[:, :, 0].argsort()][np.diag_indices(len(tmp_labels))]
    sorted_labels = sorted_labels[:,:,:-1]
    print(sorted_labels)

    images_validation = np.load("matrices_validation.npy")

    labels_validation = np.load("labels_validation.npy")
    labels_validation = labels_validation[:, labels_validation[:, :, 0].argsort()][np.diag_indices(len(labels_validation))]
    labels_validation = labels_validation[:,:,:-1]

    # model.summary()

    # earlyStopping = EarlyStopping(monitor='val_loss', patience=40, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, verbose=1, min_delta=1e-1, mode='min')

    model.fit(images, labels, validation_data=(images_validation, labels_validation), epochs=1000, batch_size=64, callbacks=[mcp_save, reduce_lr_loss])

    model.save_weights(NAME_BACKBONE+".h5", overwrite="True", save_format="h5", options=None)


# model.summary()

# tf.keras.saving.save_model(model, "model.tf")
