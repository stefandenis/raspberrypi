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


speech = VoiceRecognition('commands/inainte/inainte_calculator2.wav',1024)

command = sio.loadmat('sample.mat')
sample = command['Sample']

weights = sio.loadmat('io_weights.mat')

inW = weights['inW']
outW = weights['outW']
tip = 3

speech.signal_samples()
speech.scale_samples()
t1 = ti.time()
speech.segmentation()
trun = ti.time()-t1
print("segmentation time: ",trun)

print(np.size(speech.normalized_vector))

t1 = ti.time()
RDT_matrix = speech.RDT(speech.normalized_vector, 1024,6)
trun = ti.time() - t1 
print("timp rdt fara segmentare: ",trun)

t1 = ti.time()
RDT_matrix = speech.RDT(speech.signal,1024,6)
trun = ti.time()-t1
print("timp rdt semnal util ", trun) 


'''

scores = elmPredict_optim(sample, inW, outW, tip)
label = np.argmax(scores)
print(label)

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




