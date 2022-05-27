from main import parse_opt
from model.autoencoder import autoencoder_model
from model.cnn import cnn_1d_model, cnn_2d_model
from model.dnn import dnn_model
from model.resnet import resnet_18, resnet_101, resnet_152
from model.LSTM import lstm_model

from utils.load_predict_data import test_data_2D, test_label_2D, test_data_1D, test_label_1D
from utils.tools import recall_m, precision_m, f1_m, to_onehot, r2_keras
from utils.save_data import start_save_data
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from angular_grad import AngularGrad
import argparse
import numpy as np
import os
import tensorflow as tf

opt = parse_opt()
def Predict(data, model):
  if opt.condition_train:
    train_label = to_onehot(train_label)
    test_label  = to_onehot(test_label)

  if model == 'dnn':
    data = data.reshape(len(data), int(opt.input_shape*2))
    network = dnn_model(opt)
  if model == 'cnn_1d':
    network = cnn_1d_model(opt, training=None)
  if opt.model == 'resnet_cnn_2d':
    inputs = Input(shape=[128, 128, 2])
    output = resnet_152(opt)(inputs, training=None)
    network = Model(inputs, output)
  if opt.model == 'cnn_2d':
    network = cnn_2d_model(opt, [128, 128, 2])
  if opt.model == 'autoencoder':
    network = autoencoder_model(train_data)
  if opt.model == 'lstm':
    network = lstm_model(opt)
  
  if opt.load_weight:
    if os.path.exists(os.path.join(opt.save_dir, opt.model)):
      print(f'\nLoad weight: {os.path.join(opt.save_dir, opt.model)}\n')
      network.load_weights(os.path.join(opt.save_dir, opt.model))
      
  y_pred = network.predict(data)
  return y_pred

def main():
  result = {}
  for name in test_data_1D:
    # test_data_2D, test_label_2D, test_data_1D, test_label_1D
    y_pred_1d = Predict(test_data_1D, 'lstm')
    y_pred_2d = Predict(test_data_2D, 'resnet_cnn_2d')
    y_pred = (float(y_pred_1d) + float(y_pred_2d))/2
    r2, mae_, mse_ = all_matric(test_label_1D, y_pred)
    print(f'\n-----{name}:      R2: {r2}, MAE: {mae_}, MSE: {mse}-----')
    
if __name__ == '__main__':
  main()
