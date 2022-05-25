from tensorflow.keras.layers import Conv1D, Activation, Dense, concatenate, BatchNormalization, GlobalAveragePooling1D, Input, MaxPooling1D, Lambda, GlobalAveragePooling2D, ReLU, MaxPooling2D, Flatten, Dropout, LSTM

def lstm_model(opt, input_shape):
  inputs = Input(shape=input_shape)
  x = LSTM(units=100, return_sequences=True)(x)
  x = Dropout(0.2)(x)
  x = LSTM(units=50, return_sequences=False)(x)
  x = Dropout(0.2)(x)
  x = Dense(units=opt.num_classes, activation='sigmoid')(x)
  m = Model(inputs, x, name='resnet34')
  return m
