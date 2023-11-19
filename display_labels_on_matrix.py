import numpy as np
import matplotlib.pyplot as plt

CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
NEURONS = 64


images_test = np.load("labels_test_4_n_neurons.npy")

print("Nb images : " + str(len(images_test)))


for experience in range(len(images_test)):
    # Matrix
    matrix = np.zeros((NEURONS, NEURONS))
    for i in range(len(images_test[experience])):
        print(images_test[experience])
        if images_test[experience][i][-1] == 1:
            matrix[round(images_test[experience][i][1])][round(images_test[experience][i][0])] += 1
        else:
            matrix[round(images_test[experience][i][1])][round(images_test[experience][i][0])] -= 1

    plt.imshow(matrix)
    plt.colorbar()
    plt.show()

            

