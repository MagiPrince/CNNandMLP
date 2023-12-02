import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor

SIZE_MATRIX = 64
NEURONS = 16

SIZE_SIDE = np.sqrt(np.power(SIZE_MATRIX, 2)/NEURONS)
NB_NEURON_BY_SIDE = SIZE_MATRIX / SIZE_SIDE
MEDIAN = np.median(range(round(SIZE_SIDE)))

image = np.load("matrices_training.npy")[0]

array_of_medians = np.arange(start=MEDIAN, stop=SIZE_MATRIX, step=SIZE_SIDE)

matrix_of_centers = np.zeros((round(NB_NEURON_BY_SIDE), round(NB_NEURON_BY_SIDE), 2))
for i in range(round(NB_NEURON_BY_SIDE)):
    for j in range(round(NB_NEURON_BY_SIDE)):
        image[floor(array_of_medians[j])][floor(array_of_medians[i])] = [255, 255, 255]
    for j in range(SIZE_MATRIX):
        if array_of_medians[i] < SIZE_MATRIX-array_of_medians[0]-1:
            image[j][ceil(array_of_medians[i]+array_of_medians[0])] = [255, 0, 0]
            image[ceil(array_of_medians[i]+array_of_medians[0])][j] = [255, 0, 0]


plt.imshow(image)
plt.xlabel("x")
plt.ylabel("y")
plt.show()