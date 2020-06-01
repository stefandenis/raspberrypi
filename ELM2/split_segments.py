
import sys
sys.path.append(r'C:\Users\Denis\Desktop\Licenta')
from VoiceRecognition import VoiceRecognition
from mix_columns import mix_columns
import scipy.io as sio
import numpy as np
import os

from matplotlib import pyplot as plt 



vr_set_name = "vr_train_mod_continuu_set.mat"

Samples = np.empty((16,0))
Labels = []



nr_samples = 4096

'''
    Processing the "inceput_rostire" and "final_rostire" class 


'''
os.chdir("commands")
print(os.getcwd())
count = 0
clase = ['inceput_rostire','rostire_in_curs', 'final_rostire']
for clasa in clase:
    
    commands = os.listdir()

    for command in commands:

        working_directory = os.getcwd() + "\\" + command
        os.chdir(working_directory)
        print(os.getcwd())

        recordings = os.listdir()    

        for recording in recordings:
            signal_path = os.getcwd() + "\\" + recording
            print(signal_path)
            speech = VoiceRecognition(signal_path = signal_path,window_samples = 128)
            speech.signal_samples()
            speech.scale_samples()
            speech.segmentation()

            if clasa == 'inceput_rostire':
                clasification_sample = speech.signal[0:nr_samples]
                label = 2

            if clasa == 'final_rostire':
                clasification_sample = speech.signal[-1:-(nr_samples+1):-1]
                label = 4

            if clasa == "rostire_in_curs":
                #clasification_sample = speech.signal[nr_samples:(len(speech.signal)-nr_samples)]
                sample_from_middle = len(speech.signal)//2
                clasification_sample = speech.signal[(sample_from_middle-2048):(sample_from_middle+2047)]
                label = 3           
            

            #for i in range(0,len(clasification_sample),nr_samples):

            segment_talk_sample = VoiceRecognition()
            #RDT_sts = VoiceRecognition.RDT(clasification_sample[i:i+nr_samples],128)
            RDT_sts = speech.RDT(clasification_sample,128)
            feature_vector = segment_talk_sample.feature_vector(RDT_sts)
            feature_vector_reshaped = np.reshape(feature_vector, (segment_talk_sample.nr_spectrum * segment_talk_sample.M_segments,1))
            Samples = np.append(Samples,feature_vector_reshaped,axis = 1)
            count = count + 1
            Labels.append(label)
            
    
   
        os.chdir("..")







os.chdir("../liniste")

liniste_recordings = os.listdir()
label = 1
for liniste_recording in liniste_recordings:
    signal_path = os.getcwd() + "\\" + liniste_recording
    print(signal_path)
    speech = VoiceRecognition(signal_path = signal_path,window_samples = 128)
    speech.signal_samples()
    speech.scale_samples()
    speech.segmentation()

    RDT_matrix = speech.RDT(speech.signal,128)
    feature_vector = speech.feature_vector(RDT_matrix)
    feature_vector_reshaped = np.reshape(feature_vector, (speech.nr_spectrum * speech.M_segments,1))
    Samples = np.append(Samples,feature_vector_reshaped,axis = 1)
    count = count + 1

    Labels.append(label)
    
    # plt.figure()
    # plt.subplot(211)
    # plt.xlim(0,4096)
    # plt.ylim(-1,1)
    # plt.plot(speech.signal)

    # plt.subplot(312)
    # plt.xlim(0,4096)
    # plt.ylim(-1,1)
    # plt.plot(speech.normalized_vector)

    # plt.subplot(313)
    # plt.imshow(feature_vector, cmap = "gray")

    # plt.show()
    
    
    
    
                                             


     
Labels = np.array(Labels)
Labels = np.reshape(Labels,(1,count))


(Samples,Labels) = mix_columns(Samples,Labels)

print("Samples")
print(Samples)

print("shape samples:", Samples.shape)

print("Labels")
print(Labels)

print("shape lables:", Labels.shape)



vr_set = {
    "__header__": "Continuum voice recording train set created with python 3.7",
    "Samples"   : Samples,
    "Labels"    : Labels

}
os.chdir('..')
print(vr_set)

sio.savemat(vr_set_name,vr_set)

























    



