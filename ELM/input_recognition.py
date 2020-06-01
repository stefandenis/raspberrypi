import os
import sys
sys.path.append(os.getcwd()+'/..')

import pyaudio
import wave
import numpy as np
import scipy.io as sio
import scipy.io.wavfile
from time import sleep
from elm_model import elmPredict_optim,hidden_nonlin
from VoiceRecognition import VoiceRecognition
import time as ti
import keyboard

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 8192 # 2^12 samples for buffer
record_secs = 2 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii)
wav_output_filename = 'test2.wav' # name of .wav file
M_segments = 6

audio = pyaudio.PyAudio() # create pyaudio instantiation
audio.get_default_input_device_info()
sleep(0.5)
# create pyaudio stream



stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)
print("recording")
frames = []
# loop through stream and append audio chunks to frame array
for ii in range(0,int((samp_rate/chunk)*record_secs)):
    data = stream.read(chunk)
    frames.append(data)
print("finished recording")
# stop the stream, close it, and terminate the pyaudio instantiation
stream.stop_stream()

stream.close()
audio.terminate()
# save the audio frames as .wav file
wavefile = wave.open(wav_output_filename,'wb')
wavefile.setnchannels(chans)
wavefile.setsampwidth(audio.get_sample_size(form_1))
wavefile.setframerate(samp_rate)
wavefile.writeframes(b''.join(frames))
wavefile.close() 
speech = VoiceRecognition(signal_path = "test2.wav",window_samples = 1024)
speech.signal_samples()
speech.scale_samples()
speech.segmentation()
RDT_matrix = speech.RDT(speech.signal, speech.window_samples,M_segments)
feature_vector = speech.feature_vector(RDT_matrix)
feature_vector_reshaped = np.reshape(feature_vector, (speech.nr_spectrum*speech.M_segments,1))
'''
Making of sampels, each column represents an example, a recording of a spoken utterance 
Each row represents the component of that specific example or we can refer to them as the
number of input neurons
'''
Sample = feature_vector_reshaped
io_weights = sio.loadmat("io_weights.mat")
inW = io_weights["inW"]
outW = io_weights["outW"]
tip = 3
t1 = ti.time()
scores = elmPredict_optim(Sample, inW, outW, tip)
trun = ti.time()-t1
print( " prediction time: %f seconds" %trun)
print("scores: ",scores)
command_list = ['avarii', 'claxon', 'frana','hey_jarvis', 'inainte', 'inapoi', 'lumini', 'radio', 'start', 'stop']
maximum = np.max(scores)
print(np.shape(scores))

for i in range(np.size(scores)):
    if scores[i][0] == maximum:
        print(command_list[i])
    
