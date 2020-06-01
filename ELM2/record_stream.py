import sys
import os
sys.path.append(os.getcwd() + '/..')

from VoiceRecognition import VoiceRecognition
import numpy as np 
from matplotlib import pyplot as plt 
import pyaudio
import time
import struct
import wave
from elm_model import elmPredict_optim,hidden_nonlin
import time as ti
import scipy.io as sio



# ELM 
io_weights_command = sio.loadmat("../ELM/io_weights.mat")
inW_command = io_weights_command["inW"]
outW_command = io_weights_command["outW"]



# ELM2
io_weights = sio.loadmat("elm_continuu_weights.mat")
inW = io_weights["inW"]
outW = io_weights["outW"]

tip = 3
M_segments_ELM2 = 4
M_segments_ELM = 6

sound_list = ['liniste','rostire']
command_list = ['avarii', 'claxon', 'frana','hey_jarvis', 'inainte', 'inapoi', 'lumini', 'radio', 'start', 'stop']


LINISTE = 0
ROSTIRE = 1


chans = 1
form_1 = pyaudio.paInt16
samp_rate = 44100

p = pyaudio.PyAudio()

wav_output_filename = 'continum.wav' # name of .wav file

command_frames = np.empty((0,1))


previous_label = LINISTE
previous_frame = 0





def callback(in_data, frame_count, time_info, status):
    #print(in_data)
    global previous_label
    global command_frames
    global previous_frame
    frames = []
    #print(len(in_data))
    #print("the whole data pack: ", in_data)
    for i in range(0,len(in_data),2):
        #frames.append(struct.unpack('<h',in_data[i:i+2])[0])
        frames.append(struct.unpack('<h',in_data[i:i+2])[0])
        #print(struct.unpack('<h',in_data[i:i+2])[0])
    #print(frames)
    #print(len(frames))
    frames = np.array(frames)
    window = VoiceRecognition(samples = frames)
    window.scale_samples()
    window.segmentation()
    RDT_matrix = window.RDT(window.signal, 128,M_segments_ELM2)
    feature_vector = window.feature_vector(RDT_matrix)
    Sample = np.reshape(feature_vector, (window.nr_spectrum*window.M_segments,1))
    
    scores = elmPredict_optim(Sample, inW, outW, tip)
    label = np.argmax(scores)
    current_label = label 
    #print(sound_list[label])
    #print("current_label:",current_label)
    #print("previous_label", previous_label)
        

    if current_label == ROSTIRE and previous_label == LINISTE:
        #print(np.shape(command_frames))
        #print(np.shape(window.signal))
        command_frames = window.signal
        command_frames = np.append(command_frames, previous_frame, axis = 0)
       
    if current_label == ROSTIRE and previous_label == ROSTIRE:
        command_frames = np.append(command_frames, window.signal, axis = 0)

    if previous_label == ROSTIRE and current_label == LINISTE:
        command_frames = np.append(command_frames, window.signal, axis = 0)
        #print("am ajuns aici")
        command = VoiceRecognition() 
        try:
            RDT_matrix = command.RDT(command_frames, 1024,M_segments_ELM)
            feature_vector = command.feature_vector(RDT_matrix)
            Sample = np.reshape(feature_vector, (command.nr_spectrum*command.M_segments,1))  

            #print(np.shape(Sample))   
            scores = elmPredict_optim(Sample, inW_command, outW_command, tip)
            print(command_list[np.argmax(scores)])
        except ValueError:
            print("Secventa vocala a fost mult prea scurta")

        # plt.figure()

        # plt.subplot(211)
        # plt.ylim(-1,1)
        # plt.plot(command_frames)
        
        # plt.subplot(212)
        # plt.imshow(feature_vector, cmap = 'gray')
        # plt.show()

        command_frames = np.empty((1,0))
        
    previous_frame = window.signal
    #print(previous_frame)
        
    
    #print(scores)
    #print(np.shape(scores))
    #print(command_list[maxim])
             
    
    #np.append(frames,np.array([struct.unpack('<h',in_data[i:i+2])[0]]),axis = 0)
    #frames = struct.unpack('<h',in_data)[0]
    #print(frames)
    previous_label = label
    
    return (in_data, pyaudio.paContinue)



stream = p.open(
            format = pyaudio.paInt16,
            channels = chans,
            rate = 44100,
            frames_per_buffer = 4096,
            input = True,
            output = False,
            stream_callback =  callback) 






stream.start_stream()


time.sleep(1000)
stream.close()

