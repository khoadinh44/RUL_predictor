import numpy as np
import os
from keras import backend as K
import pandas as pd
import pickle as pkl
import pywt
import noisereduce as nr
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from utils.extract_features import extracted_feature_of_signal
from sklearn.metrics import r2_score, accuracy_score
from os import path
import scipy
from sklearn.cluster import KMeans

def hankel_svdvals(data, hankel_window_size, slice_window_size):
    """ Slices data in 'slice_window_size' and compute hankel matrix singular values with 'hankel_window_size' """

    n_slices = len(data)//slice_window_size
    hankel_svd = []

    for i in range(n_slices):
        sample_data = data[slice_window_size*i : slice_window_size*(i+1)]
        c = sample_data[0: hankel_window_size]
        r = sample_data[hankel_window_size - 1: ]
        h = scipy.linalg.hankel(c, r)

        hankel_svd.append(scipy.linalg.svdvals(h))

    return hankel_svd

def correlation_coeffs(data, baseline, norm_interval, filter_window_size, filter_polyorder):
    """ Normalizes data, select baseline data and compute the correlation coefficients """

    a, b = norm_interval
    diff = b - a

    MIN = min([min(x) for x in data])
    MAX = max([max(x) for x in data])
    DIFF = MAX - MIN

    data_norm = [diff*(x - MIN)/(DIFF) + a for x in data]
    
    x = data_norm[baseline]
    
    R = [sum(np.multiply(x, y))/(np.sqrt(sum(x**2)*sum(y**2))) for y in data_norm]

    # Passing savgol filter
    return scipy.signal.savgol_filter(R, filter_window_size, filter_polyorder)

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
  new_label = np.zeros((len(label), int(np.max(label))))
  for idx, i in enumerate(label):
    new_label[idx][int(i-1.)] = 1.
  return new_label

def back_onehot(label):
  a = []
  for i in label:
    a.append(np.argmax(i)+1)
  return np.array(a)

def r2_keras(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  K.sum(K.square( y_pred - K.mean(y_true) ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return SS_res/SS_tot 

def r2_numpy(y_true, y_pred):
    """Coefficient of Determination 
    """
    SS_res =  np.sum(( y_pred - np.mean(y_true) )**2)
    SS_tot = np.sum(( y_true - np.mean(y_true) )**2)
    return SS_res/SS_tot 

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

def rmse(y_true, y_pred):
    return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean())

def all_matric(y_true_rul, y_pred_rul, y_true_con, y_pred_con):
    y_true_rul = np.squeeze(y_true_rul)
    y_pred_rul = np.squeeze(y_pred_rul)
    y_true_con = np.squeeze(y_true_con)
    y_pred_con = np.squeeze(y_pred_con)
    
    acc = accuracy_score(y_true_con, y_pred_con)
    r2 = r2_score(y_true_rul, y_pred_rul)
    mae_ = mae(y_true_rul, y_pred_rul)
    rmse_ = rmse(y_true_rul, y_pred_rul)
    return r2, mae_, rmse_, acc
    
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

def extract_feature_image(df, opt, type_data, feature_name='horiz accel'):
    DATA_POINTS_PER_FILE=2560
    WIN_SIZE = 20
    WAVELET_TYPE = 'morl'
    if feature_name == 'horiz accel':
        data = df[4]
    else:
        data = df[5]
        
    if type_data == '2d':
        data = np.array([np.mean(data[i: i+WIN_SIZE]) for i in range(0, DATA_POINTS_PER_FILE, WIN_SIZE)])
        coef, _ = pywt.cwt(data, np.linspace(1, 128, 128), WAVELET_TYPE)
        # transform to power and apply logarithm?!
        coef = np.log2(coef**2 + 0.001)
        coef = (coef - coef.min())/(coef.max() - coef.min())
    else:
        coef = data 
    return coef

def denoise(signals):
    all_signal = []
    for x in signals:
        L1, L2, L3 = pywt.wavedec(x, 'coif7', level=2)
        all_ = np.expand_dims(np.concatenate((L1, L2, L3)), axis=0)
        if all_signal == []:
          all_signal = all_
        else:
          all_signal = np.concatenate((all_signal, all_))
        # all_signal.append(nr.reduce_noise(y=x, sr=2559, hop_length=20, time_constant_s=0.1, prop_decrease=0.5, freq_mask_smooth_hz=25600))
    return np.expand_dims(all_signal, axis=-1)

def convert_to_image(name_bearing, opt, type_data, num_files, time=None):
    data = {'x': [], 'y': []}
    if type_data == '2d':
      print('-'*10, f'Convert to 2D data', '-'*10, '\n')
    else:
      print('-'*10, f'Maintain 1D data', '-'*10, '\n')
    
    for i in range(num_files):
        name = f"/acc_{str(i+1).zfill(5)}.csv"
        file_ = os.path.join(opt.main_dir_colab, name_bearing)+name
        # print(file_)
        if path.exists(file_):
            df = pd.read_csv(file_, header=None)
            coef_h = np.expand_dims(extract_feature_image(df, opt, type_data, feature_name='horiz accel'), axis=-1)
            coef_v = np.expand_dims(extract_feature_image(df, opt, type_data, feature_name='vert accel'), axis=-1)
            x_ = np.concatenate((coef_h, coef_v), axis=-1).tolist()
            y_ = gen_rms(coef_h)
            data['x'].append(x_)
            data['y'].append(y_)
            
    if time != None: # Create label for test set -----------------------------------------
        data['y'] = np.ones_like(data['y'])
        data['y'][time: ] = np.linspace(1, 0, len(data['y'][time: ]))
    else:  # Create label for training set -----------------------------------------
        window = 50
        label_new = hankel_svdvals(data['y'], 25, window)
        label_new = correlation_coeffs(label_new, 0, [-1, 1], 3, 2)
        
        kmeans = KMeans(n_clusters=2, random_state=0).fit(label_new.reshape(-1, 1))
        thres = np.array(kmeans.labels_).astype(np.int32) > 0.5
        PSDclean = label_new * thres
        
        unique, counts = np.unique(PSDclean, return_counts=True)
        normal_time = counts[0]*window
        label_train = np.ones_like(data['y'])
        label_train[normal_time:] = np.linspace(1, 0, len(label_train[normal_time:]))
        data['y'] = label_train
        
    if type_data=='extract':
      print('-'*10, 'Convert to Extracted data', '-'*10, '\n')
      hor_data = extracted_feature_of_signal(np.array(data['x'])[:, :, 0])
      ver_data = extracted_feature_of_signal(np.array(data['x'])[:, :, 1])
      data_x = np.concatenate((hor_data, ver_data), axis=-1)
      data['x'] = data_x
    
    if opt.scaler == 'MinMaxScaler':
      scaler = MinMaxScaler
    if opt.scaler == 'MaxAbsScaler':
      scaler = MaxAbsScaler
    if opt.scaler == 'StandardScaler':
      scaler = StandardScaler
    if opt.scaler == 'RobustScaler':
      scaler = RobustScaler
    if opt.scaler == 'Normalizer':
      scaler = Normalizer
    if opt.scaler == 'QuantileTransformer':
      scaler = QuantileTransformer
    if opt.scaler == 'PowerTransformer':
      scaler = PowerTransformer
      
    if opt.scaler != None:
      hor_data = np.array(data['x'])[:, :, 0]
      ver_data = np.array(data['x'])[:, :, 1]
      print('-'*10, f'Use scaler: {opt.scaler}', '-'*10, '\n')
      if opt.scaler == 'FFT':
        hor_data = np.expand_dims(FFT(hor_data), axis=-1)
        ver_data = np.expand_dims(FFT(ver_data), axis=-1)
      elif opt.scaler == 'denoise':
        hor_data = denoise(hor_data)
        ver_data = denoise(ver_data)
      else:
        hor_data = scaler_transform(hor_data, scaler)
        ver_data = scaler_transform(ver_data, scaler)
      data_x = np.concatenate((hor_data, ver_data), axis=-1)
      data['x'] = data_x
    else:
      print('-'*10, 'Raw data', '-'*10, '\n')
      data['x'] = np.array(data['x'])
#     data['y'], _ = fit_values(2.31e-5, 0.99, 1.10, 1.68e-93, 28.58, np.array(data['y']))
    # data['y'] = convert_1_to_0(data['y'])

    x_shape = data['x'].shape
    y_shape = data['y'].shape
    print(f'Train data shape: {x_shape}   Train label shape: {y_shape}\n')
    return data

def FFT(signals):
  fft_data = []
  for signal in signals:
    signal = np.fft.fft(signal)
    signal = np.abs(signal) / len(signal)
    signal = signal[range(signal.shape[0] // 2)]
    signal = np.expand_dims(signal, axis=0)
    if fft_data == []:
      fft_data = signal
    else:
      fft_data = np.concatenate((fft_data, signal))
  return fft_data

# ---------------------- Load_predict_data.py----------------------
def seg_data(data, length):
  all_data = {}
  num=0
  for name in length:
    all_data[name] = data[num: num+length[name]]
    num += length[name]
  return all_data

def percent_error(y_true, y_pred):
    y_pred = y_pred.reshape(-1, )
    E = 100.*(y_true - y_pred)/y_true
    E = E.astype(np.float32)
    A = []
    for i in E:
        if i <= 0.:
            A.append(np.exp(-np.log(0.5)*(i/5.)))
        else:
            A.append(np.exp(np.log(0.5)*(i/20.)))
    score = []
    for i in range(1, 12):
        score.append(A[i])
    return np.mean(score)

# ----------------------Creating label----------------------

def fit_values(k, b, Y, M, B, rrms):  # fit_values(2.31e-5, 0.99, 1.10, 1.68e-93, 28.58, rrms_1)
    y_array = []
    x = []
    for i in range(len(rrms)):
        x.append(i)
        y = Y + M*pow(i,B)
        if(y < 1.1):
            y = k*x[i] + b
        y_array.append(y)
    return y_array, x

def gen_rms(col):
    return np.squeeze(np.sqrt(np.mean(col**2)))

def convert_1_to_0(data):
    f_data = (data - np.min(data))/(np.max(data) - np.min(data))
    return 1. - f_data
