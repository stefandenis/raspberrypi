import scipy.io as sio
import numpy as np
import os

import sys
sys.path.append(os.getcwd()+'\..')


from VoiceRecognition import VoiceRecognition
from matplotlib import pyplot as plt 
from mix_columns import mix_columns

vr_set_name = "vr_train_set.mat"
working_directory = os.getcwd() + "\commands"
M_segments = 6


os.chdir(working_directory)
print(os.getcwd())

commands = os.listdir()
count = 0
Label = 1 
Labels = []
for command in commands:
    
    working_directory = os.getcwd() + "\\" + command
    os.chdir(working_directory)
    
    
    ''' 
    This block is used to process the wav files from a certain command folder and put 
    the result in a .mat file  
    '''
    voice_recordings = os.listdir()
    
    for voice_recording in voice_recordings:
        
        signal_path = os.getcwd() + "\\" + voice_recording
        print(signal_path)
        speech = VoiceRecognition(signal_path = signal_path,window_samples = 1024)
        speech.signal_samples()
        speech.scale_samples()
        speech.segmentation()
          
        RDT_matrix = speech.RDT(speech.signal, speech.window_samples,M_segments)
        feature_vector = speech.feature_vector(RDT_matrix)
        print(speech.nr_spectrum*speech.M_segments)
        feature_vector_reshaped = np.reshape(feature_vector, (speech.nr_spectrum*speech.M_segments,1))
        print(count)
        
        '''
        Making of sampels, each column represents an example, a recording of a spoken utterance 
        Each row represents the component of that specific example or we can refer to them as the
        number of input neurons
        '''
        Sample = feature_vector_reshaped
        if count == 0:
            Samples = np.zeros((speech.nr_spectrum * speech.M_segments , 1))
            Samples = Samples + Sample
            
        else: 
            Samples = np.append(Samples,Sample, axis = 1)
            
        
        
        count = count + 1

        '''
        Labels for commands 
        1-avarii 
        2-claxon
        3-frana
        4-hey jarvis 
        5-inainte
        6-inapoi
        7-lumini
        8-radio
        9-start
        10-stop
        '''
        Labels.append(Label)  
        
    Label += 1

    

    '''
    end of processing block 
    '''
    print("shape of samples:",np.shape(Samples))
    print("Samples",Samples)
    os.chdir('..')


Labels = np.array(Labels)
Labels = np.reshape(Labels,(1,count))
print("Shape of labels: ", np.shape(Labels))
print("Labels: ",Labels)

if vr_set_name == "vr_train_set.mat":
    (Samples,Labels) = mix_columns(Samples,Labels)

vr_set = {
    "__header__": "Voice recording train set created with python 3.7",
    "Samples"   : Samples,
    "Labels"    : Labels

}
os.chdir('..')
print(vr_set)

sio.savemat(vr_set_name,vr_set)