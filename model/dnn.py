from functools import partial
import keras
import tensorflow as tf
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, Dropout
import keras.backend as K
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
