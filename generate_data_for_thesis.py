import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from resnet_and_mlp import resnetModelWithLocalization
from qresnet_and_mlp import qresnetModelWithLocalization
from evaluate_model_classic import evaluate_model_classic, qevaluate_model_classic

def custom_loss(y_true, y_pred):
    # Coordinates loss
    coords_loss = tf.keras.losses.mean_squared_error(y_true[:, :, :2], y_pred[:, :, :2])
    
    # Confidence loss
    confidence_loss = tf.keras.losses.binary_crossentropy(y_true[:, :, 2:], y_pred[:, :, 2:])
    
    # Combine both losses
    return coords_loss + confidence_loss

NB_NEURONS = 64
NAME_BASE = "resnet18_cnn_64n_x_y_conf_corrected_final"
NAME_WEIGHTS = "qresnet18_cnn_"+str(NB_NEURONS)+"n_x_y_conf_corrected_final_8_bits_train_from_base"

images = np.load("matrices_training.npy")

labels = np.load("labels_training_4_n_neurons.npy")

images_validation = np.load("matrices_validation.npy")

labels_validation = np.load("labels_validation_4_n_neurons.npy")

images_test = np.load("matrices_test.npy")

labels_test = np.load("labels_test_4_n_neurons.npy")

# print(labels_test[0])

array_epochs = [2, 5, 10, 20, 30, 50, 100, 300, 500]#, 750, 1000]
array_epochs_b4_evaluation = [2, 3, 5, 10, 10, 20, 50, 200, 200]#, 250, 250]

dict_results = {}

# Loop of the number of models we are going to train to get training and results data
for iteration in range(0, 5):
    print("--------------------- Experience " + str(iteration) + " ---------------------")
    # Creating model for this experience
    tf.keras.utils.set_random_seed((iteration+1)*17*100)
    model = qresnetModelWithLocalization(NB_NEURONS)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=custom_loss, metrics=['accuracy'])

    model.load_weights(NAME_BASE+".h5", skip_mismatch=False, by_name=False, options=None)

    # model.summary()

    # Setting or creating variables for this experience
    dict_results[str(iteration)] = {}
    loss_history = []
    val_loss_history = []
    for i, nb_epochs in enumerate(array_epochs_b4_evaluation):
        print("--------------------- Working on epochs " + str(array_epochs[i]) + " ---------------------")
        dict_results[str(iteration)][str(array_epochs[i])] = {}

        if i != 0:
            model.load_weights(NAME_WEIGHTS+".h5", skip_mismatch=False, by_name=False, options=None)

        model.fit(images, labels, validation_data=(images_validation, labels_validation), epochs=nb_epochs, batch_size=64)

        model.save_weights(NAME_WEIGHTS+".h5", overwrite="True", save_format="h5", options=None)

        true_positif, false_positif, false_negative, f1_score, prediction_dict, nb_gt = qevaluate_model_classic(images_test, labels_test, NAME_WEIGHTS, NB_NEURONS)
        dict_results[str(iteration)][str(array_epochs[i])]["values_computed"] = [true_positif, false_positif, false_negative, f1_score, nb_gt]
        dict_results[str(iteration)][str(array_epochs[i])]["prediction"] = prediction_dict


        loss_history += model.history.history['loss']
        val_loss_history += model.history.history['val_loss']
        

    dict_results[str(iteration)]['loss'] = loss_history
    dict_results[str(iteration)]['val_loss'] = val_loss_history

np.save(NAME_WEIGHTS+'_0_5_05_095.npy', dict_results)

# acc = model.history.history['val_accuracy']
# print(acc) # [0.9573, 0.9696, 0.9754, 0.9762, 0.9784]

