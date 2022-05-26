from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras.models import Model

def lstm_model(opt):
  inputs = Input(shape=[opt.input_shape, 2])
  x = LSTM(units=10, return_sequences=True)(inputs)
  x = tf.keras.activations.tanh(x)
  x = Dropout(0.2)(x)
  x = LSTM(units=50, return_sequences=True)(x)
  x = tf.keras.activations.tanh(x)
  x = Dropout(0.2)(x)
  x = LSTM(units=1, return_sequences=False)(x)
  x = tf.keras.activations.sigmoid(x)
  # l1 = Dropout(0.2)(x)
  # x = Dense(512, activation='tanh')(x)
  # x = Dropout(0.2)(x)
  # x = concatenate([l1, x])
  # x = Dense(units=opt.num_classes, activation='sigmoid')(x)
  m = Model(inputs, x)
  return m
