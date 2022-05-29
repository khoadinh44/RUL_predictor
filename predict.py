from train import parse_opt
from model.autoencoder import autoencoder_model
from model.cnn import cnn_1d_model, cnn_2d_model
from model.dnn import dnn_model
from model.resnet import resnet_18, resnet_101, resnet_152, resnet_50
from model.LSTM import lstm_model

from utils.load_predict_data import test_data_2D, test_label_2D, test_data_1D, test_label_1D
from utils.tools import all_matric
from utils.save_data import start_save_data
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from angular_grad import AngularGrad
import argparse
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter("ignore")
tf.get_logger().setLevel('INFO')
import logging
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True

opt = parse_opt()
def Predict(data, model):
  if opt.condition_train:
    train_label = to_onehot(train_label)
    test_label  = to_onehot(test_label)

  if model == 'dnn':
    data = (data[:, :, 0], data[:, :, 1])
    network = dnn_model(opt)
  if model == 'cnn_1d':
    network = cnn_1d_model(opt, training=None)
  if model == 'resnet_cnn_2d_50':
    inputs = Input(shape=[128, 128, 2])
    output = resnet_50(opt)(inputs, training=None)
    network = Model(inputs, output)
  if model == 'cnn_2d':
    network = cnn_2d_model(opt, [128, 128, 2])
  if model == 'autoencoder':
    network = autoencoder_model(train_data)
  if model == 'lstm':
    network = lstm_model(opt)
  
  print(f'\nLoad weight: {os.path.join(opt.save_dir, model)}\n')
  network.load_weights(os.path.join(opt.save_dir, model))
      
  y_pred = network.predict(data)
  return y_pred

def main():
  result = {}
  for name in test_data_1D:
    # test_data_2D, test_label_2D, test_data_1D, test_label_1D
    print(f'\nShape 1D data: {test_data_1D[name].shape}')
    print(f'Shape 2D data: {test_data_2D[name].shape}')
    y_pred_1d = Predict(test_data_1D[name], 'dnn')
    y_pred_2d = Predict(test_data_2D[name], 'resnet_cnn_2d_50')
    print(f'\nShape 1D prediction: {y_pred_1d.shape}')
    print(f'Shape 2D prediction: {y_pred_2d.shape}')

    y_pred = (y_pred_1d+y_pred_2d)/2.

    plt.plot(test_label_1D[name], c='b')
    plt.plot(y_pred, c='r')
    plt.title(f'{name}: combination prediction.')
    plt.savefig(f'{name}_all.png')
    plt.close()

    plt.plot(test_label_1D[name], c='b')
    plt.plot(y_pred_1d, c='r')
    plt.title(f'{name}: dnn prediction.')
    plt.savefig(f'{name}_dnn.png')
    plt.close()

    plt.plot(test_label_1D[name], c='b')
    plt.plot(y_pred_2d, c='r')
    plt.title(f'{name}: Resnet 50 prediction.')
    plt.savefig(f'{name}_resnet_50.png')
    plt.close()
    r2, mae_, mse_ = all_matric(test_label_1D[name], y_pred)
    print(f'\n-----{name}:      R2: {r2}, MAE: {mae_}, MSE: {mse_}-----')
    
if __name__ == '__main__':
  warnings.filterwarnings("ignore", category=FutureWarning)
  main()

