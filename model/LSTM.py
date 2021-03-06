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

def lstm_model(opt, training=None, inputs=None):
  if opt.mix_model==False:
    inputs = Input(shape=[opt.input_shape, 2])
  x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001))(inputs)
  x = BatchNormalization()(x, training=training)
  x = Activation('relu')(x)
  x = MaxPooling1D(pool_size=4, strides=None)(x)

  for i in range(3):
    x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i, training=training)

  x = MaxPooling1D(pool_size=4, strides=None)(x)

  for i in range(8):
    x = identity_block(x, kernel_size=3, filters=96, stage=2, block=i, training=training)

  x = MaxPooling1D(pool_size=4, strides=None)(x)

  for i in range(36):
    x = identity_block(x, kernel_size=3, filters=192, stage=3, block=i, training=training)

  x = MaxPooling1D(pool_size=4, strides=None)(x)

  for i in range(3):
    x = identity_block(x, kernel_size=3, filters=384, stage=4, block=i, training=training)
  x = tf.keras.layers.Bidirectional(LSTM(units=512, return_sequences=False))(x)

  if opt.mix_model:
      return x
  x = Dense(units=opt.num_classes, activation='sigmoid')(x)
  m = Model(inputs, x)
  return m

def lstm_extracted_model(opt, training=None, inputs=None):
  x = tf.keras.layers.Bidirectional(LSTM(128, activation='relu', 
                return_sequences=False, 
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5)))(inputs)
#   x = BatchNormalization()(x, training=training)
  return x
