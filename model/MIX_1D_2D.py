from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras.models import Model
from tensorflow_addons.layers import MultiHeadAttention
from keras import layers, regularizers
import keras.backend as K

def TransformerLayer(q, v, k, num_heads=4, training=None):
    q = tf.keras.layers.Dense(256,   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(q)
    k = tf.keras.layers.Dense(256,   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(k)
    v = tf.keras.layers.Dense(256,   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                     bias_regularizer=regularizers.l2(1e-4),
                                     activity_regularizer=regularizers.l2(1e-5))(v)
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    ma  = MultiHeadAttention(head_size=num_heads, num_heads=num_heads)([q, k, v]) + k
    ma = BatchNormalization()(ma, training=training)
    ma = Activation('relu')(ma)
    ma = Dropout(0.1)(ma, training=training)
    return ma

def mix_model(opt, cnn_1d_model, resnet_50, lstm_extracted_model, input_1D, input_2D, input_extracted, training=False):
  out_1D = Dropout(0.1)(cnn_1d_model(opt, training, input_1D), training=training)
  out_2D = Dropout(0.1)(resnet_50(opt)(input_2D, training=training), training=training)
  out_extracted = lstm_extracted_model(opt, training, input_extracted)
  
  network_1D = Model(input_1D, out_1D, name='network_1D')
  network_2D = Model(input_2D, out_2D, name='network_2D')
  network_extracted = Model(input_extracted, out_extracted, name='network_extracted')
  
  hidden_out_1D = network_1D([input_1D])
  hidden_out_2D = network_2D([input_2D])
  hidden_out_extracted = network_extracted([input_extracted])
  
  merged_value_0 = TransformerLayer(hidden_out_1D, hidden_out_2D, hidden_out_extracted, 12, training)
  merged_value_1 = concatenate([hidden_out_1D, hidden_out_2D, hidden_out_extracted], axis=-1, name='merged_value_1')
    
  Condition = Dense(3, 
                    activation='softmax', 
                    name='Condition', 
                    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5))(merged_value_1)
  RUL = Dense(1, 
              activation='sigmoid', 
              name='RUL', 
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5))(merged_value_0)
  return Condition, RUL
  
  
