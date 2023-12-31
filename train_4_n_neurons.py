from model import customModelWithLocalization
from resnet_and_mlp import resnetModelWithLocalization
from qresnet_and_mlp import qresnetModelWithLocalization
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

NAME_BACKBONE = "cnn_test_4_n"
TRAIN = True

images = np.load("matrices_training.npy")

labels = np.load("labels_training_4_n_neurons.npy")

model = resnetModelWithLocalization(16)

def coordinates_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true[:, :, :2], y_pred[:, :, :2])

def confidence_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true[:, :, 2:], y_pred[:, :, 2:])

def custom_loss(y_true, y_pred):
    # Coordinates loss
    coords_loss = tf.keras.losses.mean_squared_error(y_true[:, :, :2], y_pred[:, :, :2])
    
    # Confidence loss
    confidence_loss = tf.keras.losses.binary_crossentropy(y_true[:, :, 2:], y_pred[:, :, 2:])
    
    # Combine both losses
    return coords_loss + confidence_loss

# Compile the model with separate loss functions for each output
# model.compile(optimizer='adam', loss=[coordinates_loss, confidence_loss], metrics=['accuracy'])

if os.path.isfile(NAME_BACKBONE+".h5") and not TRAIN:

    images_test = np.load("matrices_test.npy")

    labels_test = np.load("labels_test_4_n_neurons.npy")

    model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)

    # Get predictions using the model
    pred = model.predict(images_test[4:5])[0]
    print(pred.shape)
    print(pred)
    lab = labels_test[4:5][0]
    print(lab)

    # model.summary()
    plt.imshow(images_test[4:5][0])
    plt.show()

else:
    # Train model
    # model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=['accuracy'])
    model.compile(optimizer=Adam(), loss=custom_loss, metrics=['accuracy'])

    images_validation = np.load("matrices_validation.npy")

    labels_validation = np.load("labels_validation_4_n_neurons.npy")

    # model.summary()

    earlyStopping = EarlyStopping(monitor='loss', patience=400, verbose=0, mode='min')
    mcp_save_val_loss_min = ModelCheckpoint('val_loss_min_4_n_neurons.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')
    mcp_save_loss_min = ModelCheckpoint('loss_min_4_n_neurons.h5', save_best_only=True, save_weights_only=True, monitor='loss', mode='min')
    mcp_save_val_accuracy_max = ModelCheckpoint('val_accuracy_max_4_n_neurons.h5', save_best_only=True, save_weights_only=True, monitor='val_accuracy', mode='max')
    mcp_save_accuracy_max = ModelCheckpoint('accuracy_max_4_n_neurons.h5', save_best_only=True, save_weights_only=True, monitor='accuracy', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=500, verbose=1, mode='min')

    model.fit(images, labels, validation_data=(images_validation, labels_validation), epochs=500, batch_size=64, callbacks=[mcp_save_val_loss_min, mcp_save_loss_min, mcp_save_val_accuracy_max, mcp_save_accuracy_max])

    model.save_weights(NAME_BACKBONE+".h5", overwrite="True", save_format="h5", options=None)


# model.summary()

# tf.keras.saving.save_model(model, "model.tf")
