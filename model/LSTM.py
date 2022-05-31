from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras.models import Model
from keras import layers, regularizers
import keras.backend as K


def identity_block(input_tensor, kernel_size, filters, stage, block, training):
    conv_name_base = 'res' + str(stage) + str(block) + '_branch'
    bn_name_base = 'bn' + str(stage) + str(block) + '_branch'

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=training)
    x = Activation('relu')(x)
#     x = Dropout(0.2)(x)

    x = Conv1D(filters,
               kernel_size=kernel_size,
               strides=1,
               padding='same',
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5),
              name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x, training=training)

    if input_tensor.shape[2] != x.shape[2]:
        x = layers.add([x, Lambda(lambda y: K.repeat_elements(y, rep=2, axis=2))(input_tensor)])
    else:
        x = layers.add([x, input_tensor])

    x = BatchNormalization()(x, training=training)
    x = Activation('relu')(x)
#     x = Dropout(0.2)(x)
    return x

def lstm_model(opt, training=None):
  inputs = Input(shape=[opt.input_shape, 2])
  x = LSTM(units=12, return_sequences=True)(inputs)
  x = LSTM(units=24, return_sequences=True)(x)
  x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = MaxPooling1D(pool_size=4, strides=None)(x)

  for i in range(2):
    x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i, training=training)

  x = MaxPooling1D(pool_size=4, strides=None)(x)

  for i in range(2):
    x = identity_block(x, kernel_size=3, filters=96, stage=2, block=i, training=training)

  x = MaxPooling1D(pool_size=4, strides=None)(x)

  for i in range(2):
    x = identity_block(x, kernel_size=3, filters=192, stage=3, block=i, training=training)

  x = MaxPooling1D(pool_size=4, strides=None)(x)

  for i in range(2):
    x = identity_block(x, kernel_size=3, filters=384, stage=4, block=i, training=training)

  x = GlobalAveragePooling1D()(x)  
  x = Dense(units=384, activation='relu')(x)
  x = Dropout(0.2)(x)
  x = Dense(units=opt.num_classes, activation='sigmoid')(x)
  m = Model(inputs, x)
  return m
