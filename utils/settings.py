import glob
import os

## Settings to choose
idx_GRU = 1                             # 1, 2, or 3
data_type = ''             # 'DG_data', 'FV_data', 'Fourier_data'
ianscluster = False
flag_load_only_few_files = True

# Settings for data
if ianscluster:
    base_path = '/usr/local/storage/wenzeltn/A_final_data_marius/data/' + data_type   # ianscluster path
else:
    # base_path = '/usr/local/storage/wenzeltn/A_final_data_marius/data/' + data_type  # ianscluster path
    base_path = '/home/martin/Documents/Thesis/SDKN/datasets' + data_type        # anm03 path


# Model parameters
depth = 0  # amount of residual blocks
kernel = 1  # kernel size of conv layers
n_hidden_1 = 32  # number of hidden neurons 1
n_hidden_2 = 64  # number of hidden neurons 2

# Learning parameters
batch_size = 32
 # batch_size used during training
num_epochs_nn = 50  # number of training epochs
num_epochs_sdkn = 10  # number of training epochs
val_split = 0.01  # percentage of training data used as validation data, between (0,1)

# Learning rate
initialLearningRate = 0.001  # initial learning rate
decayRate = 0.5  # exponential decay rate
decayEpochs_nn = 5  # decay steps
decayEpochs_sdkn = 10  # decay steps
doStairCase = True  # Use stepwise instead of continuous exp. decay
