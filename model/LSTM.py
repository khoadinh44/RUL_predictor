from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras.models import Model

def lstm_model(opt):
  inputs = Input(shape=[opt.input_shape, 2])
  x = LSTM(units=100, return_sequences=True)(inputs)
  x = tf.keras.activations.tanh(x)
  x = Dropout(0.2)(x)
  x = LSTM(units=50, return_sequences=True)(x)
  x = tf.keras.activations.tanh(x)
  x = Dropout(0.2)(x)
  x = GlobalAveragePooling1D(data_format='channels_first', keepdims=False)(x)
  x = Dense(1024, activation='tanh')(x)
  x = Dropout(0.5)(x)
  x = Dense(units=opt.num_classes, activation='sigmoid')(x)
  m = Model(inputs, x)
  return m
