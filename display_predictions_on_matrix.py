import tensorflow as tf
from tensorflow import keras
from model import customModelWithLocalization
from resnet_and_mlp import resnetModelWithLocalization
from qresnet_and_mlp import qresnetModelWithLocalization
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

NAME_BACKBONE = "cnn_and_mlp_4_n_neurons"
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
NEURONS = 16


images_test = np.load("matrices_test.npy")

print("Nb images : " + str(len(images_test)))

labels_test = np.load("labels_test.npy")[0]

model = resnetModelWithLocalization(NEURONS)

if not os.path.isfile(NAME_BACKBONE+".h5"):
    sys.exit(1)

model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)

# Get predictions using the model
results = model.predict(images_test)

# Matrix
matrix = np.zeros((NEURONS, 64, 64))

for i in range(len(results)):
    detection_in_range= 0
    true_detection = 0

    for j in range(len(results[i])):
        if round(results[i][j][0]) < 50:
            matrix[j][round(results[i][j][1])][round(results[i][j][0])] += 1

# print(matrix)

for i in range(NEURONS):
    plt.imshow(matrix[i])
    plt.colorbar()
    plt.show()

            

