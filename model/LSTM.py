from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras.models import Model

def lstm_model(opt):
  inputs = Input(shape=[opt.input_shape, 2])
  x1 = LSTM(units=512, return_sequences=True)(inputs)
  x1 = LSTM(units=256, return_sequences=False)(x1)
  fc = Dense(512, activation='relu')(x1)
  fc = Dense(256, activation='relu')(fc) + x1
  # fc = Dropout(0.2)(fc)
#   x = concatenate([x, fc])
  fc = Dense(units=opt.num_classes, activation='sigmoid')(fc)
  m = Model(inputs, fc)
  return m
