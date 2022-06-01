from model.autoencoder import autoencoder_model
from model.cnn import cnn_1d_model, cnn_2d_model
from model.dnn import dnn_model
from model.resnet import resnet_18, resnet_101, resnet_152, resnet_50
from model.LSTM import lstm_model
from utils.tools import recall_m, precision_m, f1_m, to_onehot, r2_keras
from utils.save_data import start_save_data
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from angular_grad import AngularGrad
import argparse
import numpy as np
import os
import tensorflow as tf
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--input_shape', default=1279, type=int, help='1279 for using fft, 2559 for raw data')
    parser.add_argument('--num_classes', default=1, type=str, help='class condition number: 3, class rul condition: 1')
    parser.add_argument('--model', default='cnn_2d', type=str, help='lstm, dnn, cnn_1d, resnet_cnn_2d, cnn_2d, autoencoder')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--scaler', default=None, type=str)
    parser.add_argument('--main_dir_colab', default=None, type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--condition_train', default=False, type=bool)
    parser.add_argument('--rul_train', default=True, type=bool)
    parser.add_argument('--load_weight', default=False, type=bool)
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt, train_data, train_label, test_data, test_label):
  print(f'Shape of train set: {train_data.shape}  {train_label.shape}')
  print(f'Shape of test set: {test_data.shape}  {test_label.shape}')
  if opt.condition_train:
      train_label = to_onehot(train_label)
      test_label  = to_onehot(test_label)
  val_data, val_label = test_data[:1000], test_label[:1000]

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
  
  if opt.load_weight:
    if os.path.exists(os.path.join(opt.save_dir, opt.model)):
      print(f'\nLoad weight: {os.path.join(opt.save_dir, opt.model)}\n')
      network.load_weights(os.path.join(opt.save_dir, opt.model))
      

  if opt.condition_train:
    network.compile(optimizer=AngularGrad(), loss='categorical_crossentropy', metrics=['acc', f1_m, precision_m, recall_m]) # loss='mse'
  if opt.rul_train:
    network.compile(optimizer=AngularGrad(1e-4), loss=tf.keras.losses.MeanSquaredLogarithmicError(), metrics=['mae', r2_keras, tf.keras.metrics.mean_squared_error], run_eagerly=True) # https://keras.io/api/losses/
  network.summary()
  history = network.fit(train_data, train_label,
                      epochs     = opt.epochs,
                      batch_size = opt.batch_size,
                      validation_data = (val_data, val_label),
                      # callbacks = [callbacks]
                      )
  # optimizer='rmsprop'
  if opt.condition_train:
      _, test_acc,  test_f1_m,  test_precision_m,  test_recall_m  = network.evaluate(test_data, test_label, verbose=0)
      print(f'----------Score in test set: \n Accuracy: {test_acc}, F1: {test_f1_m}, Precision: {test_precision_m}, recall: {test_recall_m}' )
  if opt.rul_train:
      _, test_mae, test_r2, test_mse = network.evaluate(test_data, test_label, verbose=0)
      print(f'----------Score in test set: \n mae: {test_mae}, r2: {test_r2}, mse: {test_mse}' )
  network.save(os.path.join(opt.save_dir, opt.model))

if __name__ == '__main__':
  opt = parse_opt()
  start_save_data(opt)
  if opt.condition_train:
    from utils.load_condition_data import train_data, train_label, test_data, test_label
    main(opt, train_data, train_label, test_data, test_label)
  if opt.rul_train:
    from utils.load_rul_data import train_data_rul, train_label_rul, test_data_rul, test_label_rul
    main(opt, train_data_rul, train_label_rul, test_data_rul, test_label_rul)
