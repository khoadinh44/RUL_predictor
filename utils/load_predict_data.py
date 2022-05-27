import os
import numpy as np
import pywt
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
from train import parse_opt
from utils.tools import load_df, save_df, df_row_ind_to_data_range, extract_feature_image, convert_to_image

DATA_POINTS_PER_FILE = 2560
TIME_PER_REC = 0.1
SAMPLING_FREQ = 25600 # 25.6 KHz
SAMPLING_PERIOD = 1.0/SAMPLING_FREQ

WIN_SIZE = 20
WAVELET_TYPE = 'morl'
VAL_SPLIT = 0.1

opt = parse_opt()
np.random.seed(1234)

main_dir_colab = opt.main_dir_colab
train_main_dir = main_dir_colab + 'Learning_set/'
test_main_dir = main_dir_colab + 'Test_set/'

test_data_path_2D   = load_df(main_dir_colab + 'test_data_rul.pkz')
test_label_path_2D  = load_df(main_dir_colab + 'test_label_rul.pkz')

test_data_path_1D   = load_df(main_dir_colab + 'test_data_1D.pkz')
test_label_path_1D  = load_df(main_dir_colab + 'test_label_1D.pkz')

length = {'Bearing1_3': 1802,
         'Bearing1_4': 1327,
         'Bearing1_5': 2685,
         'Bearing1_6': 2685,
         'Bearing1_7': 1752,
         'Bearing2_3': 1202,
         'Bearing2_4': 713,
         'Bearing2_5': 2337,
         'Bearing2_6': 572,
         'Bearing2_7': 200,
         'Bearing3_3': 410}
