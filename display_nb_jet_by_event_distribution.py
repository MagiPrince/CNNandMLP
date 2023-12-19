import h5py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

path_jet = "data/jet/"
path_jet ="../keras-cv-yolov8-quantized/simulated_data/jets"


########################################################################
# Jets
########################################################################

dir_files_jet = os.listdir(path_jet)
dir_files_jet = sorted(dir_files_jet)

array_nb_elements = np.zeros(30)

for folder_element, jet_file in enumerate(dir_files_jet):
    print("--------------------- file_jet " + str(jet_file) + " ---------------------")

    f = h5py.File(os.path.join(path_jet, jet_file), "r")

    dset = f["caloCells"]

    # Creating a dictionnary of type parameter : index
    data = dset["2d"]
    array_indexes_to_convert = list(data.dtype.fields.keys())
    dict_data_jet = {array_indexes_to_convert[i]: i for i in range(len(array_indexes_to_convert))}

    data_jet = dset["2d"][()]

    f.close()

    for i in range(len(data_jet)):
        df_jet = pd.DataFrame(data_jet[i],  columns=list(dict_data_jet.keys()))

        array_nb_elements[df_jet["AntiKt4EMTopoJets_eta"].count()] += 1

print(array_nb_elements)

plt.bar(np.arange(0, 30, 1), array_nb_elements)
plt.show()