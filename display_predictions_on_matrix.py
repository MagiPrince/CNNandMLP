from colorsys import hls_to_rgb
import tensorflow as tf
from tensorflow import keras
from model import customModelWithLocalization
from resnet_and_mlp import resnetModelWithLocalization
from qresnet_and_mlp import qresnetModelWithLocalization
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

NAME_BACKBONE = "cnn_test"
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
NEURONS = 16
SIZE_MATRIX = 64


images_test = np.load("matrices_test.npy")

print("Nb images : " + str(len(images_test)))

model = resnetModelWithLocalization(NEURONS)

if not os.path.isfile(NAME_BACKBONE+".h5"):
    sys.exit(1)

model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)

# Get predictions using the model
results = model.predict(images_test)

# Matrix
matrix = np.zeros((64, 64, 3), dtype=int)


colors = [hls_to_rgb(2/3 * i/(NEURONS-1), 0.5, 0.5) for i in range(NEURONS) ]

for i in range(len(results)):
    detection_in_range= 0
    true_detection = 0

    for j in range(NEURONS):
        if results[i][j][-1] > 0.0:
            # if j == 9:
            #     print(results[i][j])
            #     print([int(colors[j][0]*255), int(colors[j][1]*255), int(colors[j][2]*255)])
            matrix[round(results[i][j][1])][round(results[i][j][0])] = [int(colors[j][0]*255), int(colors[j][1]*255), int(colors[j][2]*255)]

# for j in range(SIZE_MATRIX):
#     if array_of_medians[i] < SIZE_MATRIX-array_of_medians[0]-1:
#         image[j][ceil(array_of_medians[i]+array_of_medians[0])] = [255, 0, 0]
#         image[ceil(array_of_medians[i]+array_of_medians[0])][j] = [255, 0, 0]

# print(matrix)

# for i in range(NEURONS):
plt.imshow(matrix)

legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label="D.B. : " + str(i+1)) for i in range(NEURONS)]

plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

plt.xlabel("x")
plt.ylabel("y")
plt.show()

            

