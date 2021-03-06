from train import parse_opt
from model.autoencoder import autoencoder_model
from model.cnn import cnn_1d_model, cnn_2d_model
from model.resnet import resnet_18, resnet_101, resnet_152, resnet_50
from model.MIX_1D_2D import mix_model
from model.LSTM import lstm_extracted_model, lstm_model
from model.MIX_1D_2D import mix_model
from utils.load_predict_data import test_data_2D , test_data_1D , test_data_extract , test_data_c, test_label_1D
from utils.tools import all_matric, back_onehot
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
  if model == 'mix':
    input_extracted = Input((14, 2), name='Extracted_LSTM_input')
    input_1D = Input((opt.input_shape, 2), name='LSTM_CNN1D_input')
    input_2D = Input((128, 128, 2), name='CNN_input')
    Condition, RUL = mix_model(opt, lstm_model, resnet_101, lstm_extracted_model, input_1D, input_2D, input_extracted, False)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=[Condition, RUL])
  name = f'model_{opt.condition}'
  print(f'\nLoad weight: {os.path.join(opt.save_dir, name)}\n')
  network.load_weights(os.path.join(opt.save_dir,  f'model_{opt.condition}'))
      
  Condition, RUL = network.predict(data)
  return Condition, RUL 

def main():
  result = {}
  for name in test_data_1D:
    # test_data_2D, test_label_2D, test_data_1D, test_label_1D
    print(f'\nShape 1D data: {test_data_1D[name].shape}')
    print(f'Shape 2D data: {test_data_2D[name].shape}')
    Condition, RUL = Predict([test_data_1D[name], test_data_2D[name], test_data_extract[name]], 'mix')

    plt.plot(test_label_1D[name], c='b')
    plt.plot(RUL, c='r')
    plt.title(f'{name}: combination prediction.')
    plt.savefig(f'{name}_all.png')
    plt.close()
    Condition = back_onehot(Condition)
    r2, mae_, mse_, acc = all_matric(test_label_1D[name], RUL, test_data_c[name], Condition)
    acc = round(acc*100, 4)
    mae_ = round(mae_, 4)
    rmse_ = round(mse_, 4)
    r2 = round(r2, 4)
    print(f'\n-----{name}:      R2: {r2}, MAE: {mae_}, RMSE: {rmse_}, Acc: {acc}-----')
    
if __name__ == '__main__':
  warnings.filterwarnings("ignore", category=FutureWarning)
  main()
