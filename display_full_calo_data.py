import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# calocell_file = "RealData/calocellD3PD_2018Data.h5"
# jet_file = "RealData/jetD3PD_2018Data.h5"

calocell_file = "data/calocell/user.cantel.33075755._000001.calocellD3PD_mc16_JZW4.r14423.h5"
jet_file = "data/jet/user.cantel.33075755._000001.jetD3PD_mc16_JZW4.r14423.h5"

########################################################################
# Calorimeter
########################################################################

f = h5py.File(calocell_file, "r")

# dset = f["caloCells"]

# Creating a dictionnary of type parameter : index
# data = dset["2d"]
# array_indexes_to_convert = list(data.dtype.fields.keys())
# dict_data_calo = {array_indexes_to_convert[i]: i for i in range(len(array_indexes_to_convert))}

data_calo = f["caloCells"]["2d"][("cell_eta", "cell_phi")][0]

f.close()

# df_calo = pd.DataFrame(data_calo[0],  columns=["cell_eta", "cell_phi"])

########################################################################
# Plot
########################################################################

# Creation of Dataframes pandas
df_calo = pd.DataFrame(data_calo, columns=["cell_eta", "cell_phi"])
df_calo = df_calo[df_calo['cell_eta'].abs() < 2.4]

plt.plot(df_calo['cell_eta'], df_calo['cell_phi'], linewidth=1, alpha=1)
plt.xlabel("eta")
plt.ylabel("phi")
plt.show()
