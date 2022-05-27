from functools import partial
import keras
import tensorflow as tf
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.layers import LSTM, Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout
import keras.backend as K
from keras import layers, regularizers
from keras.models import Model
from tensorflow.keras.applications import DenseNet121


def TransformerLayer(x=None, c=48, num_heads=4*3):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    q   = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
    k   = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
    v   = Dense(c, use_bias=True, 
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                  bias_regularizer=regularizers.l2(1e-4),
                  activity_regularizer=regularizers.l2(1e-5))(x)
    ma  = MultiHeadAttention(head_size=c, num_heads=num_heads)([q, k, v]) 
    ma = Dropout(0.5)(ma)
    return ma

# For m34 Residual, use RepeatVector. Or tensorflow backend.repeat
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
    x = Dropout(0.2)(x)

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
    x = Dropout(0.2)(x)
    return x
  
def cnn_1d_model(opt, training=None):
    '''
    The model was rebuilt based on the construction of resnet 34 and inherited from this source code:
    https://github.com/philipperemy/very-deep-convnets-raw-waveforms/blob/master/model_resnet.py
    '''
    inputs = Input(shape=[opt.input_shape, 2])
    x = LSTM(units=12, return_sequences=True)(inputs)
    x = Conv1D(48,
               kernel_size=80,
               strides=4,
               padding='same',
               kernel_initializer='glorot_uniform',
               kernel_regularizer=regularizers.l2(l=0.0001),)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=4, strides=None)(x)


    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=48, stage=1, block=i, training=training)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(4):
        x = identity_block(x, kernel_size=3, filters=96, stage=2, block=i, training=training)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(23):
        x = identity_block(x, kernel_size=3, filters=192, stage=3, block=i, training=training)

    x = MaxPooling1D(pool_size=4, strides=None)(x)

    for i in range(3):
        x = identity_block(x, kernel_size=3, filters=384, stage=4, block=i, training=training)

    x = GlobalAveragePooling1D()(x)
    
    # x = TransformerLayer(x, c=384*2)

    if opt.rul_train:
      x = Dense(opt.num_classes, activation='sigmoid')(x)
    if opt.condition_train:
      x = Dense(opt.num_classes, activation='softmax')(x)

    m = Model(inputs, x, name='resnet34')
    return m

def cnn_2d_model(opt, input_shape=[128, 128, 2]):
  DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")
  model = keras.models.Sequential([
            DefaultConv2D(filters=64, kernel_size=7, input_shape=input_shape), 
            Activation('relu'), 
            MaxPooling2D(pool_size=4), 
            Dropout(0.5), 
            DefaultConv2D(filters=128), 
            Activation('relu'), 
            MaxPooling2D(pool_size=4),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dropout(0.5)])
  if opt.rul_train:
    model.add(Dense(units=opt.num_classes, activation='sigmoid'))
  if opt.condition_train:
    model.add(Dense(units=opt.num_classes, activation='softmax'))
  return model
