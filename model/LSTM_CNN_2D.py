from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras.models import Model
from keras import layers, regularizers
import keras.backend as K

def mix_model(opt, model_1D, model_2D, input_1D, input_2D, training)
#   input_1D = Input((opt.input_shape, 2), name='lstm_input')
#   input_2D = Input((128, 128, 2), name='CNN_input')
  
  out_1D = model_1D(opt, training, input_1D)
  out_2D = model_2D(opt)(input_2D, training=training)
  
  network_1D = Model(input_1D, out_1D)
  network_2D = Model(input_2D, out_2D)
  
  hidden_out_1D = network_1D([input_1D])
  hidden_out_2D = network_2D([input_2D])
  
  merged_hidden = concatenate([hidden_out_1D, hidden_out_2D], axis=-1, name='merged_hidden_layer')
  output = Dense(1, activation='sigmoid')(merged_hidden)
  return output
  
  
