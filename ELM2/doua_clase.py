
import sys
import os
sys.path.append(os.getcwd() + '\..')
from VoiceRecognition import VoiceRecognition
from mix_columns import mix_columns
import scipy.io as sio
import numpy as np
import os

from matplotlib import pyplot as plt 

LINISTE_LABEL = 1
ROSTIRE_LABEL = 2

vr_set_name = "vr_train_liniste_rostire_set.mat"

Samples = np.empty((16,0))
Labels = []
label = ROSTIRE_LABEL
M_segments = 4

nr_samples = 2048

window = 4096

'''
    Processing the "inceput_rostire" and "final_rostire" class 


'''
print(os.getcwd())
os.chdir("../ELM/commands")
print(os.getcwd())
count = 0

enough_commands = 0   
commands = os.listdir()
for command in commands:
    enough_commands +=1 
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
        
        
        # normalized_vector_cut = speech.normalized_vector[speech.res[0][0]:speech.res[0][-1]]
        # #random_window = np.random.randint(len(normalized_vector_cut)//4096)
        # middle_window = len(speech.signal)//2
        # clasification_sample = speech.signal[int(middle_window-(window/2)):int(middle_window+(window/2))]
        #print(len(clasification_sample))
        #for i in range(0,len(clasification_sample),nr_samples):
        
        random_window = np.random.randint(round(len(speech.signal)/2048))
        clasification_sample = speech.signal[(random_window*2048):(random_window*2048+2048)]
        


        segment_talk_sample = VoiceRecognition()
        #RDT_sts = VoiceRecognition.RDT(clasification_sample[i:i+nr_samples],128)
        RDT_sts = segment_talk_sample.RDT(clasification_sample, 128, M_segments)
        feature_vector = segment_talk_sample.feature_vector(RDT_sts)
        feature_vector_reshaped = np.reshape(feature_vector, (segment_talk_sample.nr_spectrum * segment_talk_sample.M_segments,1))
        Samples = np.append(Samples,feature_vector_reshaped,axis = 1)
        count = count + 1
        Labels.append(label)
        
        # print(feature_vector)
        # plt.figure()
        # plt.subplot(311)
        # plt.ylim(-1,1)
        # plt.plot(clasification_sample)

        # plt.subplot(312)
        # plt.imshow(RDT_sts, cmap = 'gray')
        
        # plt.subplot(313)
        # plt.imshow(feature_vector, cmap = 'gray')

        # plt.show()

    os.chdir("..")

    if enough_commands == 4:
        break
    





print(os.getcwd())


os.chdir("../../ELM2/liniste")

liniste_recordings = os.listdir()
label = LINISTE_LABEL
for liniste_recording in liniste_recordings:
    signal_path = os.getcwd() + "\\" + liniste_recording
    print(signal_path)
    speech = VoiceRecognition(signal_path = signal_path,window_samples = 128)
    
    speech.signal_samples()
    speech.scale_samples()
    speech.segmentation()

    RDT_matrix = speech.RDT(speech.signal,128,M_segments)
    feature_vector = speech.feature_vector(RDT_matrix)
    feature_vector_reshaped = np.reshape(feature_vector, (speech.nr_spectrum * speech.M_segments,1))
    Samples = np.append(Samples,feature_vector_reshaped,axis = 1)
    count = count + 1
    Labels.append(label)
    
    # print(feature_vector)
    # plt.figure()
    # plt.subplot(311)
    # plt.ylim(-1,1)
    # plt.plot(speech.normalized_vector)

    # plt.subplot(312)
    # plt.imshow(RDT_matrix, cmap = 'gray')
        
    # plt.subplot(313)
    # plt.imshow(feature_vector, cmap = 'gray')

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

























    



