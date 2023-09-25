# Last Modified: 2021.08.31
import numpy as np
import torch
import librosa
from scipy.stats.stats import pearsonr 
from scipy import signal

def cal_snr(clean,enhanced,dtype='numpy'):
    if dtype == 'numpy':
        noise = enhanced - clean
        noise_pw = np.dot(noise,noise)
        signal_pw = np.dot(clean,clean)
        SNR = 10*np.log10(signal_pw/noise_pw)
    else:
        noise = enhanced - clean
        noise_pw = torch.sum(noise*noise,1)
        signal_pw = torch.sum(clean*clean,1)
        SNR = torch.mean(10*torch.log10(signal_pw/noise_pw)).item()
    return round(SNR,3)

def cal_rmse(clean,enhanced,dtype='numpy'):
    if dtype == 'numpy':
        RMSE = np.sqrt(((enhanced - clean) ** 2).mean())
    else:
        RMSE = torch.sqrt(torch.mean(torch.square(enhanced - clean))).item()
    return round(RMSE,6)
def cal_prd(clean,enhanced,dtype='numpy'):
    if dtype == 'numpy':
         PRD = np.sqrt(np.sum((enhanced - clean) ** 2) / np.sum(clean ** 2)) * 100
    else:
        PRD = torch.mul(torch.sqrt(torch.div(torch.sum(torch.square(enhanced - clean)),torch.sum(torch.square(clean)))),100).item()
    return round(PRD,3)

def cal_R2(clean,enhanced):
    R2 = pearsonr(clean,enhanced)[0]**2
    return round(R2,3)

def cal_CC(clean,enhanced):
    CC = np.correlate(clean,enhanced)[0]
    return round(CC,3)

def cal_ARV(emg):
  win = 1000
  ARV = []
  emg = abs(emg)
  for i in range(0,emg.shape[0],win):
    ARV.append((emg[i:i+win]).mean())
  return np.array(ARV)

def cal_KR(x):
  bins = np.linspace(-5,5,1001)
  pdf, _ = np.histogram(normalize(x),bins,density=True) # _ is bin
  cdf= np.cumsum(pdf)/np.sum(pdf)
  KR = (find_nearest(cdf,0.975)-find_nearest(cdf,0.025))/(find_nearest(cdf,0.75)-find_nearest(cdf,0.25))-2.91
  bin_centers = 0.5*(bins[1:] + bins[:-1])
  return KR

def cal_MF(emg,stimulus):
  # 10 - 500Hz mean frequency
  freq = librosa.fft_frequencies(sr=1000,n_fft=256)
  start = next(i for i,v in enumerate(freq) if v >=10)
  freq = np.expand_dims(freq[start:],1)
  spec = make_spectrum(emg,feature_type=None)[0][start:,:]
  weighted_f = np.sum(freq*spec,0)
  spec_column_pow = np.sum(spec,0)
  MF = weighted_f / spec_column_pow
  MF = [MF[i] for i,v in enumerate(stimulus[::32]) if v>0]
  return np.array(MF)

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def normalize(x):
  return (x-x.mean())/np.std(x)

def make_spectrum(y=None, is_slice=False, feature_type='logmag', mode=None, FRAMELENGTH=None,
                 SHIFT=None, _max=None, _min=None):
    
    D = librosa.stft(y,center=True, n_fft=256, hop_length=32, win_length=128,window=signal.hamming)
    utt_len = D.shape[-1]
    phase = np.exp(1j * np.angle(D))
    D = np.abs(D)

    ### Feature type
    if feature_type == 'logmag':
        Sxx = np.log1p(D)
    elif feature_type == 'lps':
        Sxx = np.log10(D**2)
    else:
        Sxx = D
    ### normalizaiton mode
    if mode == 'mean_std':
        # mean = np.mean(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))
        # std = np.std(Sxx, axis=1).reshape(((hp.n_fft//2)+1, 1))+1e-12
        # Sxx = (Sxx-mean)/std  
        Sxx = normalize(Sxx)  #meaningless
    elif mode == 'minmax':
        _min = np.max(Sxx)
        _max = np.min(Sxx)
        Sxx = (Sxx - _min)/(_max - _min)

    return Sxx, phase, len(y)