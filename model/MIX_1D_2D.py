from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras.models import Model
from keras import layers, regularizers
import keras.backend as K

def TransformerLayer(q, k, v, num_heads=4, training=None):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    ma  = MultiHeadAttention(head_size=c, num_heads=num_heads)([q, k, v]) 
    ma = Activation('relu')(ma)
    return ma

def mix_model(opt, cnn_1d_model, resnet_50, lstm_extracted_model, lstm_condition_model, input_1D, input_2D, input_extracted, input_type, training=False):
  out_1D = cnn_1d_model(opt, training, input_1D)
  out_2D = resnet_50(opt)(input_2D, training=training)
  out_extracted = lstm_extracted_model(opt, training, input_extracted)
  out_type = lstm_condition_model(opt, training, input_type)
  
  network_1D = Model(input_1D, out_1D, name='network_1D')
  network_2D = Model(input_2D, out_2D, name='network_2D')
  network_extracted = Model(input_extracted, out_extracted, name='network_extracted')
  network_type = Model(input_type, out_type, name='network_type')
  
  hidden_out_1D = network_1D([input_1D])
  hidden_out_2D = network_2D([input_2D])
  hidden_out_extracted = network_extracted([input_extracted])
  hidden_out_type = network_type([input_type])
  
  merged_value = concatenate([hidden_out_extracted, hidden_out_type], axis=-1, name='merged_value_layer')
  
  output = TransformerLayer(hidden_out_1D, hidden_out_2D, merged_value, 4, training)
  output = Dense(1, activation='sigmoid')(merged_hidden)
  return output
  
  
