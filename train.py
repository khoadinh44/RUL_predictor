from model.autoencoder import autoencoder_model
from model.cnn import cnn_1d_model
from model.MIX_1D_2D import mix_model
from model.resnet import resnet_18, resnet_101, resnet_152, resnet_50
from model.LSTM import lstm_extracted_model, lstm_model
from utils.tools import recall_m, precision_m, f1_m, to_onehot, r2_keras, to_onehot
from utils.save_data import start_save_data
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from angular_grad import AngularGrad
import tensorflow_addons as tfa
import argparse
import numpy as np
import os
import tensorflow as tf
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_shape', default=2559, type=int, help='1279 for using fft, 2559 for raw data')
    parser.add_argument('--num_classes', default=1, type=str, help='class condition number: 3, class rul condition: 1')
    parser.add_argument('--model', default='cnn_2d', type=str, help='mix, lstm, dnn, cnn_1d, resnet_cnn_2d, cnn_2d, autoencoder')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--data_type', default=['2d'], type=str, help='shape of data. They can be 1d, 2d, extract')
    parser.add_argument('--condition', default=None, type=str, help='c_1, c_2, c_3, c_all')
    parser.add_argument('--scaler', default=None, type=str)
    parser.add_argument('--main_dir_colab', default=None, type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--rul_train', default=True, type=bool)
    parser.add_argument('--mix_model', default=True, type=bool)
    parser.add_argument('--load_weight', default=False, type=bool)
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt, train_data_rul_1D, train_label_rul_1D, test_data_rul_1D, test_label_rul_1D, train_data_rul_2D, test_data_rul_2D, train_data_rul_extract, test_data_rul_extract, train_c, test_c):
  train_c = to_onehot(train_c)
  test_c = to_onehot(test_c)
  val_data_1D, val_data_2D, val_extract, val_c, val_label_RUL = test_data_rul_1D[:1000], test_data_rul_2D[:1000], test_data_rul_extract[:1000], test_c[:1000], test_label_rul_1D[:1000]
  val_data = [val_data_1D, val_data_2D, val_extract]
  val_label = [val_c, val_label_RUL]

  if opt.model == 'dnn':
    train_data = [train_data[:, :, 0], train_data[:, :, 1]]
    val_data, val_label = [test_data[:1000][:, :, 0], test_data[:1000][:, :, 1]], test_label[:1000]
    test_data = [test_data[:, :, 0], test_data[:, :, 1]]
    network = dnn_model(opt)
  if opt.model == 'cnn_1d':
    network = cnn_1d_model(opt, training=True)
  if opt.model == 'resnet_cnn_2d':
    # horirontal------------
    inputs = Input(shape=[128, 128, 2])
    output = resnet_50(opt)(inputs, training=True)
    network = Model(inputs, output)
  if opt.model == 'cnn_2d':
    network = cnn_2d_model(opt, [128, 128, 2])
  if opt.model == 'autoencoder':
    network = autoencoder_model(train_data)
  if opt.model == 'lstm':
    network = lstm_model(opt, training=True)
  if opt.mix_model:
    input_extracted = Input((14, 2), name='Extracted_LSTM_input')
    input_1D = Input((2559, 2), name='LSTM_CNN1D_input')
    input_2D = Input((128, 128, 2), name='CNN_input')
    Condition, RUL = mix_model(opt, lstm_model, resnet_101, lstm_extracted_model, input_1D, input_2D, input_extracted, True)
    network = Model(inputs=[input_1D, input_2D, input_extracted], outputs=[Condition, RUL])

    # data-------------------------------
    train_data = [train_data_rul_1D, train_data_rul_2D, train_data_rul_extract]
    train_label = [train_c, train_label_rul_1D]
    test_data = [test_data_rul_1D, test_data_rul_2D, test_data_rul_extract]
    test_label = [test_c, test_label_rul_1D]
  
  if opt.load_weight:
    if os.path.exists(os.path.join(opt.save_dir, f'model_{opt.condition}')):
      name = f'model_{opt.condition}'
      print(f'\nLoad weight: {os.path.join(opt.save_dir, name)}\n')
      network.load_weights(os.path.join(opt.save_dir, f'model_{opt.condition}'))
      
  network.compile(optimizer=AngularGrad(),
                  loss=['categorical_crossentropy', tf.keras.losses.MeanSquaredLogarithmicError()], 
                  metrics=['acc', 'mae', tfa.metrics.RSquare(), tf.keras.metrics.RootMeanSquaredError()], 
                  loss_weights=[0.5, 1],
                  run_eagerly=True) # https://keras.io/api/losses/ 
  network.summary()
  history = network.fit(train_data, train_label,
                        epochs     = opt.epochs,
                        batch_size = opt.batch_size,
                        validation_data = (val_data, val_label),
                      )
  network.save(os.path.join(opt.save_dir, f'model_{opt.condition}'))
  _, _, _, Condition_acc, _, _, _, _, RUL_mae, RUL_r_square, RUL_mean_squared_error = network.evaluate(val_data, val_label, verbose=0)
  Condition_acc = round(Condition_acc, 4)
  RUL_mae = round(RUL_mae, 4)
  RUL_r_square = round(RUL_r_square, 4)
  RUL_mean_squared_error = round(RUL_mean_squared_error, 4)
  print(f'\n----------Score in test set: \n Condition acc: {Condition_acc}, mae: {RUL_mae}, r2: {RUL_r_square}, rmse: {RUL_mean_squared_error}\n' )

if __name__ == '__main__':
  opt = parse_opt()
  start_save_data(opt)
  from utils.load_rul_data import train_data_rul_1D, train_label_rul_1D, \
                                  test_data_rul_1D, test_label_rul_1D, \
                                  train_data_rul_2D, \
                                  test_data_rul_2D,\
                                  train_data_rul_extract, \
                                  test_data_rul_extract,\
                                  train_c, test_c
  main(opt, train_data_rul_1D, train_label_rul_1D, test_data_rul_1D, test_label_rul_1D, train_data_rul_2D, test_data_rul_2D, train_data_rul_extract, test_data_rul_extract, train_c, test_c)
