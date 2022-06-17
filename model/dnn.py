from functools import partial
import keras
import tensorflow as tf
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, Dropout
import keras.backend as K
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras import layers, regularizers
from keras.models import Model

def dnn_model(opt):
  input_1 = keras.layers.Input(shape=[opt.input_shape, ])
  input_2 = keras.layers.Input(shape=[opt.input_shape, ])
  fc1 = keras.layers.Dense(5092, activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(input_1)
  fc1 = Dropout(0.2)(fc1)
  fc1 = keras.layers.Dense(2048,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(fc1)
  fc1 = Dropout(0.2)(fc1)
  fc1 = keras.layers.Dense(2048,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(fc1)
  fc1 = Dropout(0.2)(fc1)
  fc1 = keras.layers.Dense(1024,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(fc1)
  fc1 = Dropout(0.2)(fc1)
  fc1 = keras.layers.Dense(512,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(fc1)
  fc1 = Dropout(0.2)(fc1)
  fc1 = keras.layers.Dense(512,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(fc1)
  fc1 = BatchNormalization()(fc1)
  fc1 = Dropout(0.2)(fc1)

  fc2 = keras.layers.Dense(5092, activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(input_2)
  fc2 = Dropout(0.2)(fc2)
  fc2 = keras.layers.Dense(2048,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(fc2)
  fc2 = Dropout(0.2)(fc2)
  fc2 = keras.layers.Dense(2048,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(fc2)
  fc2 = Dropout(0.2)(fc2)
  fc2 = keras.layers.Dense(1024,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(fc2)
  fc2 = Dropout(0.2)(fc2)
  fc2 = keras.layers.Dense(512,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(fc2)
  fc2 = BatchNormalization()(fc2)
  fc2 = Dropout(0.2)(fc2)
  x = concatenate([fc1, fc2])
  if opt.rul_train:
    output = Dense(opt.num_classes, activation='sigmoid')(x)
  if opt.condition_train:
    output = Dense(opt.num_classes, activation='softmax')(x)
  model = keras.models.Model(inputs=[input_1, input_2], outputs=[output])
  return model

# def dnn_extracted_model(opt, training, inputs):
#   x = keras.layers.Flatten()(x)
#   x = keras.layers.Dense(56, activation=tf.keras.layers.ReLU(), 
#                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                                      bias_regularizer=regularizers.l2(1e-4),
#                                      activity_regularizer=regularizers.l2(1e-5))(x)
#   x = keras.layers.Dense(112,activation=tf.keras.layers.ReLU(), 
#                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                                      bias_regularizer=regularizers.l2(1e-4),
#                                      activity_regularizer=regularizers.l2(1e-5))(x)
#   x = keras.layers.Dense(224,activation=tf.keras.layers.ReLU(), 
#                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                                      bias_regularizer=regularizers.l2(1e-4),
#                                      activity_regularizer=regularizers.l2(1e-5))(x)
#   x = keras.layers.Dense(384,activation=tf.keras.layers.ReLU(), 
#                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                                      bias_regularizer=regularizers.l2(1e-4),
#                                      activity_regularizer=regularizers.l2(1e-5))(x)
#   x = BatchNormalization()(x, training=training)
#   return x

def dnn_extracted_model(opt, training=None, inputs=None):
  x = LSTM(56, activation='relu', 
                return_sequences=True, 
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5))(inputs)
  x = LSTM(112, activation='relu', 
                return_sequences=True,
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5))(x)
  x = LSTM(256, activation='relu',
                  return_sequences=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
  x = LSTM(384, activation='relu', 
                  return_sequences=False, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
  x = Dense(384)(x) 
  x = BatchNormalization()(x, training=training)   
  return x
