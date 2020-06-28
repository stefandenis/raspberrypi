import os
import sys
sys.path.append(os.getcwd() + r'/..')

from VoiceRecognition import VoiceRecognition
from matplotlib import pyplot as plt 
import numpy as np
import scipy.io as sio
import scipy.fftpack
import scipy.io.wavfile as wavfile
import wave
import pyaudio
from elm_model import * 
import time as ti 


speech = VoiceRecognition('commands/avarii/avarii_calculator2.wav',1024)

command = sio.loadmat('sample.mat')
sample = command['Sample']

weights = sio.loadmat('io_weights_50.mat')

inW = weights['inW']
outW = weights['outW']
tip = 3

speech.signal_samples()
speech.scale_samples()
speech.segmentation()





RDT_matrix = speech.RDT(speech.signal, 1024,6)
feature_vector = speech.feature_vector(RDT_matrix) 
feature_vector_reshaped = np.reshape(feature_vector, (speech.nr_spectrum*speech.M_segments,1))
t1 = ti.time()
scores = elmPredict_optim(feature_vector_reshaped, inW, outW, tip)
tfinal = ti.time()-t1
label = np.argmax(scores)

print("timp pana la predictie: ", tfinal)

print(label)


'''
plt.subplot(211)
plt.xlabel("samples magnitude")

plt.ylim(-1,1)
plt.ylabel("time(s)")
plt.plot(sample)



plt.subplot(212)
plt.xlabel('spectrograms')
plt.ylabel('spectrum')
plt.imshow(feature_vector, cmap = 'gray')



plt.show()
'''




