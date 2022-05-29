import numpy as np
import os
from keras import backend as K
import pandas as pd
import pickle as pkl
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


#----------------------#### General ####------------------------------------------------
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
  
def accuracy_m(y_true, y_pred):
  correct = 0
  total = 0
  for i in range(len(y_true)):
      act_label = np.argmax(y_true[i]) # act_label = 1 (index)
      pred_label = np.argmax(y_pred[i]) # pred_label = 1 (index)
      if(act_label == pred_label):
          correct += 1
      total += 1
  accuracy = (correct/total)
  return accuracy

def to_onehot(label):
  new_label = np.zeros((len(label), np.max(label)+1))
  for idx, i in enumerate(label):
    new_label[idx][i] = 1.
  return new_label

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def r2_numpy(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  np.sum(( y_true - y_pred )**2)
    SS_tot = np.sum(( y_true - np.mean(y_true) )**2)
    return ( 1 - SS_res/(SS_tot + np.finfo(float).eps) )

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def mse(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()

def all_matric(y_true, y_pred):
    r2 = r2_numpy(y_true, y_pred)
    mae_ = mae(y_true, y_pred)
    mse_ = mse(y_true, y_pred)
    return r2, mae_, mse_
    
#----------------------save_data.py------------------------------------------------
def read_data_as_df(base_dir):
  '''
  saves each file in the base_dir as a df and concatenate all dfs into one
  '''
  if base_dir[-1]!='/':
    base_dir += '/'

  dfs=[]
  for f in sorted(os.listdir(base_dir)):
    if f[:3] == 'acc':
      df=pd.read_csv(base_dir+f, header=None, names=['hour', 'minute', 'second', 'microsecond', 'horiz accel', 'vert accel'])
      dfs.append(df)
  return pd.concat(dfs)

def process(base_dir, out_file):
  '''
  dumps combined dataframes into pkz (pickle) files for faster retreival
  '''
  df=read_data_as_df(base_dir)
  # assert df.shape[0]==len(os.listdir(base_dir))*DATA_POINTS_PER_FILE
  with open(out_file, 'wb') as pfile:
    pkl.dump(df, pfile)
  print('{0} saved'.format(out_file))
  print(f'Shape: {df.shape}\n')

#----------------------Load_data.py------------------------------------------------
def load_df(pkz_file):
    with open(pkz_file, 'rb') as f:
        df=pkl.load(f)
    return df

def save_df(df, out_file):
  with open(out_file, 'wb') as pfile:
    pkl.dump(df, pfile)
    print('{0} saved'.format(out_file))

def df_row_ind_to_data_range(ind):
    DATA_POINTS_PER_FILE=2559
    return (DATA_POINTS_PER_FILE*ind, DATA_POINTS_PER_FILE*(ind+1))

def scaler(signal, scale_method):
  scale = scale_method().fit(signal)
  return scale.transform(signal), scale

def scaler_transform(signals, scale_method):
  data = []
  scale = scale_method()
  for signal in signals:
    if len(signal.shape) < 2:
      signal = np.expand_dims(signal, axis=-1)
    data.append(scale.fit_transform(signal))
  return np.array(data)

def extract_feature_image(df, ind, opt, feature_name='horiz accel'):
    DATA_POINTS_PER_FILE=2559
    WIN_SIZE = 20
    WAVELET_TYPE = 'morl'
    data_range = df_row_ind_to_data_range(ind)
    data = df[feature_name].values[data_range[0]: data_range[1]]
    if opt.model in ['cnn_2d', 'resnet_cnn_2d']:
        data = np.array([np.mean(data[i: i+WIN_SIZE]) for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])
        coef, _ = pywt.cwt(data, np.linspace(1,128,128), WAVELET_TYPE)
        # transform to power and apply logarithm?!
        coef = np.log2(coef**2 + 0.001)
        coef = (coef - coef.min())/(coef.max() - coef.min()) 
    else:
        coef = data.reshape(-1, 1)
        if opt.scaler == 'MinMaxScaler':
          scaler = MinMaxScaler
        elif opt.scaler == 'MaxAbsScaler':
          scaler = MaxAbsScaler
        elif opt.scaler == 'StandardScaler':
          scaler = StandardScaler
        elif opt.scaler == 'RobustScaler':
          scaler = RobustScaler
        elif opt.scaler == 'Normalizer':
          scaler = Normalizer
        elif opt.scaler == 'QuantileTransformer':
          scaler = QuantileTransformer
        elif opt.scaler == 'PowerTransformer':
          scaler = PowerTransformer
        if opt.scaler != None:
          coef = scaler_transform(coef, scaler)
    return coef

def convert_to_image(pkz_dir, opt):
    df = load_df(pkz_dir+'.pkz')
    no_of_rows = df.shape[0]
    DATA_POINTS_PER_FILE=2559
    no_of_files = int(no_of_rows / DATA_POINTS_PER_FILE)
    print(f'pkz file length: {no_of_rows}, total subsequence data: {no_of_files}')
    
    data = {'x': [], 'y': []}
    for i in range(0, no_of_files):
        coef_h = np.expand_dims(extract_feature_image(df, i, opt, feature_name='horiz accel'), axis=-1)
        coef_v = np.expand_dims(extract_feature_image(df, i, opt, feature_name='vert accel'), axis=-1)
        x_ = np.concatenate((coef_h, coef_v), axis=-1).tolist()
#         x_ = np.array([coef_h, coef_v])
        all_nums = (no_of_files-1)
        y_ = float(all_nums-i)/float(all_nums)
        data['x'].append(x_)
        data['y'].append(y_)
    data['x']=np.array(data['x'])
    data['y']=np.array(data['y'])

    # assert data['x'].shape==(no_of_files, 128, 128, 2)
    x_shape = data['x'].shape
    y_shape = data['y'].shape
    print(f'Train data shape: {x_shape}   Train label shape: {y_shape}\n')
    return data

# ---------------------- Load_predict_data.py----------------------
def seg_data(data, length):
  all_data = {}
  num=0
  for name in length:
    all_data[name] = data[num: num+length[name]]
    num += length[name]
  return all_data

