import os
import numpy as np
import pywt
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
from train import parse_opt
from utils.tools import load_df, seg_data

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

length = {'Bearing1_3': 1802,
          'Bearing1_4': 1139,
          'Bearing1_5': 2302,
          'Bearing1_6': 2302,
          'Bearing1_7': 1502,
          'Bearing2_3': 1202,
          'Bearing2_4': 612,
          'Bearing2_5': 2002,
          'Bearing2_6': 572,
          'Bearing2_7': 172,
          'Bearing3_3': 352}

test_data_2D   = seg_data(load_df(main_dir_colab + f'test_data_2D_{opt.condition}.pkz'), length)
test_data_1D   = seg_data(load_df(main_dir_colab + f'test_data_1D_{opt.condition}.pkz'), length)
test_data_extract   = seg_data(load_df(main_dir_colab + f'test_data_extract_{opt.condition}.pkz'), length)
test_data_c   = seg_data(load_df(main_dir_colab + f'test_c_{opt.condition}.pkz'), length)

test_label_1D  = seg_data(load_df(main_dir_colab + f'test_label_1D_{opt.condition}.pkz'), length)
