import numpy as np
import glob
import re
import librosa
from scipy.io import wavfile
from os import path

#############################################################################
########### Definitions of Vocoder-Processing functions #####################
#############################################################################

######################## Analyse WaveGlow ############################

def format_mel_WAVEGLOW(wav):
  # Settings for Mel-processing with WaveGlow
  hop_length = 256 # in line with WAVEGLOW
  win_length = 1024 # in line with WAVEGLOW
  n_fft = 1024
  n_mel_channels = 80
  mel_fmin = 0.0
  mel_fmax=8000.0
  MAX_WAV_VALUE = 32768.0

  nm_MEL = "_WAVEGLOW/" + nm + ".WAVEGLOW"
  if not path.exists(nm_MEL):
    # Normalize WAV
    wav = wav / MAX_WAV_VALUE

    # Compute the mel spectrum
    mel = librosa.feature.melspectrogram(wav.astype(np.float32), sr=fs, n_fft=n_fft, power=1, win_length=win_length, hop_length=hop_length, fmin=mel_fmin, fmax = mel_fmax, n_mels=n_mel_channels)

    # Range compression
    mel = np.log(mel.clip(1e-5)).transpose()

    # Save .WAVEGLOW
    nbt = mel.shape[0]
    fp = open(nm_MEL, 'wb')
    fp.write(np.asarray(mel.shape, dtype = np.int32))
    fp.write(mel.copy(order = 'C'))
    fp.close()

  else:
    shape = tuple(np.fromfile(nm_MEL, count = 2, dtype = np.int32))
    mel = np.memmap(nm_MEL, offset = 8, dtype = np.float32, shape = shape)
    nbt = shape[0]

  Hz = nbt / lg_wav_s
  print('MEL {} {:.2f}Hz'.format(mel.shape, Hz))

#############################################################################
###########                 Main Script                 #####################
#############################################################################

# List of mel-spectro to generate
vocoders_list = ["WAVEGLOW"]

myfiles = glob.glob("_WAV/*.wav")
for nm_wav in myfiles:
  (fs, wav) = wavfile.read(nm_wav)
  lg_wav_s = len(wav) / fs
  nm = re.search('([^\/]+?)\.wav', nm_wav).group(1)
  print("{}: {}Hz {}s".format(nm, fs, lg_wav_s))

  for vocoder in vocoders_list:
    globals()["format_mel_" + vocoder](wav)