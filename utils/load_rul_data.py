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

if opt.model in ['cnn_2d', 'resnet_cnn_2d']:
  train_data_path = main_dir_colab + 'train_data_rul.pkz'
  train_label_path = main_dir_colab + 'train_label_rul.pkz'
  test_data_path = main_dir_colab + 'test_data_rul.pkz'
  test_label_path = main_dir_colab + 'test_label_rul.pkz'
else:
  train_data_path = main_dir_colab + 'train_data_1D.pkz'
  train_label_path = main_dir_colab + 'train_label_1D.pkz'
  test_data_path = main_dir_colab + 'test_data_1D.pkz'
  test_label_path = main_dir_colab + 'test_label_1D.pkz'
 
 
if os.path.exists(test_data_path):
  train_data_rul  = load_df(train_data_path)
  train_label_rul = load_df(train_label_path)
  test_data_rul   = load_df(test_data_path)
  test_label_rul  = load_df(test_label_path)
else:
  # Train data-------------------------------------------------------------------------
  Bearing1_1_path = main_dir_colab + 'Learning_set/Bearing1_1'
  Bearing1_2_path = main_dir_colab + 'Learning_set/Bearing1_2'
  Bearing2_1_path = main_dir_colab + 'Learning_set/Bearing2_1'
  Bearing2_2_path = main_dir_colab + 'Learning_set/Bearing2_2'
  Bearing3_1_path = main_dir_colab + 'Learning_set/Bearing3_1'
  Bearing3_2_path = main_dir_colab + 'Learning_set/Bearing3_2'
  print('\n Training rul data'+'-'*100)
  Bearing3_2_data = convert_to_image(Bearing3_2_path, opt)
  Bearing3_1_data = convert_to_image(Bearing3_1_path, opt)
  Bearing2_2_data = convert_to_image(Bearing2_2_path, opt)
  Bearing2_1_data = convert_to_image(Bearing2_1_path, opt)
  Bearing1_2_data = convert_to_image(Bearing1_2_path, opt)
  Bearing1_1_data = convert_to_image(Bearing1_1_path, opt)

  train_data_rul = np.concatenate((Bearing3_2_data['x'], Bearing3_1_data['x'], Bearing2_2_data['x'], Bearing2_1_data['x'], Bearing1_2_data['x'], Bearing1_1_data['x']))
  train_label_rul = np.concatenate((Bearing3_2_data['y'], Bearing3_1_data['y'], Bearing2_2_data['y'], Bearing2_1_data['y'], Bearing1_2_data['y'], Bearing1_1_data['y']))

  # Test data---------------------------------------------------------------------------
  Bearing1_3_path = main_dir_colab + 'Test_set/Bearing1_3'
  Bearing1_4_path = main_dir_colab + 'Test_set/Bearing1_4'
  Bearing1_5_path = main_dir_colab + 'Test_set/Bearing1_5'
  Bearing1_6_path = main_dir_colab + 'Test_set/Bearing1_6'
  Bearing1_7_path = main_dir_colab + 'Test_set/Bearing1_7'
  Bearing2_3_path = main_dir_colab + 'Test_set/Bearing2_3'
  Bearing2_4_path = main_dir_colab + 'Test_set/Bearing2_4'
  Bearing2_5_path = main_dir_colab + 'Test_set/Bearing2_5'
  Bearing2_6_path = main_dir_colab + 'Test_set/Bearing2_6'
  Bearing2_7_path = main_dir_colab + 'Test_set/Bearing2_7'
  Bearing3_3_path = main_dir_colab + 'Test_set/Bearing3_3'
  print('\n Test rul data'+'-'*100)
  Bearing1_3_data = convert_to_image(Bearing1_3_path, opt)
  Bearing1_4_data = convert_to_image(Bearing1_4_path, opt)
  Bearing1_5_data = convert_to_image(Bearing1_5_path, opt)
  Bearing1_6_data = convert_to_image(Bearing1_6_path, opt)
  Bearing1_7_data = convert_to_image(Bearing1_7_path, opt)
  Bearing2_3_data = convert_to_image(Bearing2_3_path, opt)
  Bearing2_4_data = convert_to_image(Bearing2_4_path, opt)
  Bearing2_5_data = convert_to_image(Bearing2_5_path, opt)
  Bearing2_6_data = convert_to_image(Bearing2_6_path, opt)
  Bearing2_7_data = convert_to_image(Bearing2_7_path, opt)
  Bearing3_3_data = convert_to_image(Bearing3_3_path, opt)

  test_data_rul = np.concatenate((Bearing1_3_data['x'], Bearing1_4_data['x'], Bearing1_5_data['x'], Bearing1_6_data['x'], Bearing1_7_data['x'], Bearing2_3_data['x'], Bearing2_4_data['x'], Bearing2_5_data['x'], Bearing2_6_data['x'], Bearing2_7_data['x'], Bearing3_3_data['x']))
  test_label_rul = np.concatenate((Bearing1_3_data['y'], Bearing1_4_data['y'], Bearing1_5_data['y'], Bearing1_6_data['y'], Bearing1_7_data['y'], Bearing2_3_data['y'], Bearing2_4_data['y'], Bearing2_5_data['y'], Bearing2_6_data['y'], Bearing2_7_data['y'], Bearing3_3_data['y']))

  save_df(train_data_rul, train_data_path)
  save_df(train_label_rul, train_label_path)
  save_df(test_data_rul, test_data_path)
  save_df(test_label_rul, test_label_path)
  
print(f'Train shape: {train_data_rul.shape}   {train_label_rul.shape}\n')  
print(f'Test shape: {test_data_rul.shape}   {test_label_rul.shape}\n')
