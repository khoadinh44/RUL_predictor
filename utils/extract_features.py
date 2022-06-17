import numpy as np

# Time ---------------------------------------------------------------------
def RMS(x):
  return np.sqrt(np.mean(x**2))

def Variance(x):
  return np.var(x)

def Peak_value(x):
  return np.max(np.abs(x))

def Crest_factor(x):
  data_PvT = Peak_value(x)
  data_RMS = RMS(x)
  return data_PvT/data_RMS + 1e-10

def Kurtosis(x):
  var = Variance(x)
  return np.sum((x-np.mean(x))**4)/(float(len(x))*(var**2)) + 1e-10

def Clearance_factor(x):
  data_PvT = Peak_value(x)
  return data_PvT/np.mean(np.sqrt(abs(x)))**2 + 1e-10

def Impulse_factor(x):
  data_PvT = Peak_value(x)
  return data_PvT/np.mean(abs(x)) + 1e-10

def shape_factor(x):
  data_RMS = RMS(x)
  return data_RMS/np.mean(abs(x)) + 1e-10

def Line_integral(x):
  a = 0.
  for i in range(len(x)-1):
    a += abs(x[i+1]-x[i])
  return a

def Peak_peak_value(x):
  return max(x)-min(x)

def Skewness(x):
  mean = np.mean(x)
  return np.mean((x-mean)**3) / (np.sqrt(np.mean((x-mean)**2)))**3 + 1e-10

# Frequency ---------------------------------------------------------------------
def Peak_FFT(x):
  r = abs(np.fft.fft(x))
  return np.max(r)

def energy_FFT(x):
  r = abs(np.fft.fft(x))
  return np.average(r)

def PSD_FFT(x):
  num = float(len(x))
  fhat = np.fft.fft(x)                 
  PSD = fhat * np.conj(fhat) / num   
  return np.max(PSD.real)

def extracted_feature_of_signal(signals):
  data = []
  for signal in signals:
    features = [RMS(signal), Variance(signal), Peak_value(signal), Crest_factor(signal), \
                Kurtosis(signal), Clearance_factor(signal), Impulse_factor(signal), \
                shape_factor(signal), Line_integral(signal), Peak_peak_value(signal), \
                Skewness(signal), Peak_FFT(signal), energy_FFT(signal), PSD_FFT(signal)]
    data.append(features)
  return np.array(data)
