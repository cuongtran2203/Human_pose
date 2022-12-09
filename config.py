import numpy as np
import os
# Path for exported data, numpy arrays
DATA_PATH ="../MP_Data"
ROOT="./data/dataset"
# Actions that we try to detect
actions = np.array(['0', '5', '6','8'])

# Thirty videos worth of data
no_sequences = 65

# Folder start
start_folder = 0
# Videos are going to be 30 frames in length
sequence_length = 30
