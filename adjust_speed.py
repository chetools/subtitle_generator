import numpy as np
import pytsmod as tsm
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import librosa

x, sr = librosa.load("test2.wav")

# x, sr = sf.read("test2.wav")
# x = x.T
# x_length = x.shape[-1]  # length of the audio sequence x.

# s_fixed = 0.8
# x_s_fixed = tsm.wsola(x, s_fixed)
# sf.write("out.wav", np.column_stack((x_s_fixed, x_s_fixed)), samplerate=sr)
