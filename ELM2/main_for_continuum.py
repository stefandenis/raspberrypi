import os
import sys
sys.path.append(os.getcwd() + '\..')
from VoiceRecognition import VoiceRecognition
from matplotlib import pyplot as plt 
import numpy as np
import scipy.io as sio
import scipy.fftpack
import scipy.io.wavfile as wavfile
import wave
import pyaudio


os.chdir('..')

speech = VoiceRecognition(os.getcwd()+ r'\ELM2\liniste\liniste0.wav')

print(sys.path)
speech.signal_samples()
speech.scale_samples()
speech.segmentation()

RDT_matrix = speech.RDT(speech.signal,128,4)
feature_vector = speech.feature_vector(RDT_matrix)

window = 4096
normalized_vector_cut = speech.normalized_vector[speech.res[0][0]:speech.res[0][-1]]



middle_window = len(speech.signal)//2
clasification_sample = speech.signal[int(middle_window-(window/2)):int(middle_window+(window/2))]




print(speech.res[0])
print(speech.res[0][-1])
print(speech.res[0][0])




# plt.figure()
# plt.subplot(311)
# plt.ylim(-1,1)
# plt.plot(speech.normalized_vector)

# plt.subplot(312)
# plt.ylim(-1,1)
# plt.plot(speech.normalized_vector[speech.res[0][0]:speech.res[0][-1]])



# plt.subplot(313)
# plt.ylim(-1,1)
# plt.plot(clasification_sample)


# plt.show()

plt.figure()
plt.subplot(311)
plt.ylim(-1,1)
plt.plot(speech.signal)

plt.subplot(312)
plt.imshow(RDT_matrix)

plt.subplot(313)
plt.imshow(feature_vector)



plt.show()





