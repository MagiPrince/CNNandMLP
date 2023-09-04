import tensorflow as tf
from tensorflow import keras
from model import customModelWithLocalization
from resnet_and_mlp import resnetModelWithLocalization
import numpy as np
import os
import sys
import copy

NAME_BACKBONE = "cnn_and_mlp"
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.1


def intersection_over_union(boxA, boxB):
    #Extract bounding boxes coordinates
    x0_A, y0_A, x1_A, y1_A = boxA
    x0_B, y0_B, x1_B, y1_B = boxB

    x1_A, y1_A = x0_A+x1_A, y0_A+y1_A
    x1_B, y1_B = x0_B+x1_B, y0_B+y1_B
    
    # Get the coordinates of the intersection rectangle
    x0_I = max(x0_A, x0_B)
    y0_I = max(y0_A, y0_B)
    x1_I = min(x1_A, x1_B)
    y1_I = min(y1_A, y1_B)
    #Calculate width and height of the intersection area.
    width_I = x1_I - x0_I 
    height_I = y1_I - y0_I

    # Handle the negative value width or height of the intersection area
    if width_I < 0 or height_I < 0:
        return 0
    # Calculate the intersection area:
    intersection = width_I * height_I
    # Calculate the union area:
    width_A, height_A = x1_A - x0_A, y1_A - y0_A
    width_B, height_B = x1_B - x0_B, y1_B - y0_B
    union = (width_A * height_A) + (width_B * height_B) - intersection
    # Calculate the IoU:
    IoU = intersection/union
    # Return the IoU and intersection box
    return IoU


images_test = np.load("matrices_test.npy")

print("Nb images : " + str(len(images_test)))

labels_test = np.load("labels_test.npy")

model = resnetModelWithLocalization(30)

if not os.path.isfile(NAME_BACKBONE+".h5"):
    sys.exit(1)

model.load_weights(NAME_BACKBONE+".h5", skip_mismatch=False, by_name=False, options=None)

# Get predictions using the model
results = model.predict(images_test)

# Confusion Matrix
true_positif = 0
false_positif = 0
false_negative = 0

ap_array_labels = []
ap_array_scores = []

for i in range(len(results)):
    detection_in_range= 0
    true_detection = 0

    coord_gt = []
    for j in range(len(labels_test[i])):
        if labels_test[i][j][0] < 45 and labels_test[i][j][0] > 5 and labels_test[i][j][1] < 59 and labels_test[i][j][1] > 5:
            coord_gt.append(copy.deepcopy(labels_test[i][j][:2].tolist()))

    for j in range(len(results[i])):
        if results[i][j][0] < 45 and results[i][j][0] > 5 and results[i][j][1] > 5 and results[i][j][1] < 59:
            detection_in_range += 1
            index_iou = -1
            best_iou = -1
            for k in range(len(coord_gt)):
                gt_box = [coord_gt[k][0]-5, coord_gt[k][1]-5, 10, 10]
                pred_box = [results[i][j][0]-5, results[i][j][1]-5, 10, 10]
                iou = intersection_over_union(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    index_iou = k
            
            if best_iou >= IOU_THRESHOLD:
                coord_gt.pop(index_iou)
                true_detection += 1
    
    true_positif += true_detection
    false_positif += detection_in_range-true_detection
    false_negative += len(coord_gt)

print("True positif : " + str(true_positif))
print("False positif : " + str(false_positif))
print("False negative : " + str(false_negative))

print("F1 score : " + str((2*true_positif)/(2*true_positif+false_positif+false_negative)))

# print("Boxes : ", labels_test["boxes"][:1])
# print("Classes : ", labels_test["classes"][:1])