import numpy as np
import os
# Path for exported data, numpy arrays
DATA_PATH ="./Dataloader/MP_Data"
ROOT="./data/dataset"
# Actions that we try to detect
actions = np.array(['vay_tay', 'dam_tay', 'dap_tay','bat_tay'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 0
