from functools import partial
import keras
import tensorflow as tf
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda
import keras.backend as K
from keras import layers, regularizers
from keras.models import Model

def dnn_model(opt):
  input_ = keras.layers.Input(shape=[opt.input_shape, ])
  x = keras.layers.Dense(1024, activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(input_)
  x = keras.layers.Dense(512,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(x)
  x = keras.layers.Dense(256,activation=tf.keras.layers.ReLU(), 
                                     kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(x)
  if opt.rul_train:
    output = Activation("linear")(x)
  if opt.condition_train:
    output = Dense(opt.num_classes, activation='softmax')(x)
  model = keras.models.Model(inputs=[input_], outputs=[output])
  return model
