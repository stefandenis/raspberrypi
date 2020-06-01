import numpy as np
from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as LA 
import math
from statistics import stdev


class VoiceRecognition:
    
    def __init__(self, signal_path = None, window_samples = None, samples = None, show_console_print = False):
        self.signal_path = signal_path
        self.window_samples = window_samples
        
        self.show_console_print = show_console_print
        
        self.samples = samples
        
    
    def signal_samples(self):
        ''' 
        Return the samples of the signal. If the signal is stereo there will be a matrix with 2 columns.
        We must choose one of them. Since the signal is recorded with one microphone on 2 channels, the 2 columns 
        will have similiar samples so we can choose any of the columns.
        
        '''
        samplerate, data = wavfile.read(self.signal_path)
        
        # only one channel wav file 
        if len(np.shape(data)) == 1:
            self.samples = data

        #2 channels wav file 
        if len(np.shape(data)) == 2:
            self.samples = data[:,1]

        return self.samples 
 
 

    def scale_samples(self):
        max_value = 32678
        self.normalized_vector = self.samples / max_value

        return self.normalized_vector

    def segmentation(self):
        # old segmentation
        # segmented_vector = []
        # for value in self.normalized_vector:
        #     if value > 0.01:
        #         segmented_vector.append(value)
        #     if value < -0.01:
        #         segmented_vector.append(value)    

        # self.final_segmented_vector = np.asarray(segmented_vector)  

        #new segmentation
        threshold = 0.01
        self.res=np.where(np.abs(self.normalized_vector) > threshold)
        #print("self.res[1]",self.res[1])
        self.signal=self.normalized_vector[self.res[0]]

        # self.signal is a column vector
        self.signal=self.signal.reshape(np.size(self.res[0]),1)
        
        return self.signal

        
    def RDT(self,signal,w, M_segments):
        
        self.M_segments = M_segments

        Nsamples = np.size(signal)
        samples_per_segment = Nsamples//self.M_segments



        spectrograms_per_segment = samples_per_segment//w + 1

        n = int(np.log2(w))
        k = n-3
        final_spectrum = np.empty((k,0))

        samples = spectrograms_per_segment * w

        if self.show_console_print:
            print("Length of signal given to the RDT: ", len(signal))
            print("Size of window for RDT: ", w)
            print("spectrograms per segment: ", spectrograms_per_segment)


        if Nsamples < 350: # we don t have enough samples after segmentation so that means it has to be quiet.
            signal = np.zeros((4096,1))
            

        
        for i in range(self.M_segments):
            if self.show_console_print == True:
                print(i)
                print(np.shape(signal))
            start_of_segment = i*samples_per_segment            
            if i == (self.M_segments-1):
                
                spectrograms_per_segment = samples_per_segment//w
                samples = spectrograms_per_segment * w 

            
            #print(len(signal[(start_of_segment):(start_of_segment+samples)]))
            matrix = np.reshape(signal[(start_of_segment):(start_of_segment+samples)],(spectrograms_per_segment,w))
            
            
            if samples == 0: # this is the case where there is only one spectrogram per segment 
                spectrograms_per_segment = 1 
                matrix = np.reshape(signal[-1:-(w+1):-1],(spectrograms_per_segment , w))   
            
            
            
            

            spectrum = np.zeros((n-3,spectrograms_per_segment))
            for i in range(0,spectrograms_per_segment):
                values = matrix[i,:]
                for k in range(0,n-3):
                    delay = 2 ** k 
                    t = np.array(range(delay,w-delay-1))
                    difus = np.abs(values[t-delay] + values[t+delay] - 2*values[t]) 
                    spectrum[k,i] = np.mean(difus)/4
                #print(spectrum)    
            final_spectrum = np.append(final_spectrum, spectrum, axis = 1)
        
        

        return final_spectrum


    
    def feature_vector(self, spectrogram):
        (nr_spectrum, nr_spectrograms) = np.shape(spectrogram)
        self.nr_spectrum = nr_spectrum
        segment_length = self.M_segments
        segments = round(nr_spectrograms / segment_length)


        self.segments = segments
        feature_vector = np.arange(segment_length * nr_spectrum,dtype=np.float64)   
        feature_vector = np.reshape(feature_vector,(nr_spectrum, segment_length))
        
      
        if self.show_console_print == True:
            print("Spectrogram : ",spectrogram)
        for i in range(nr_spectrum): 
          
            for j in range(segment_length):
                
                #print(spectrogram[i][j*segments:j*segments+segments])

                spectrogram_avg = np.mean(spectrogram[i][j*segments:j*segments+segments])
                
                feature_vector[i][j] = spectrogram_avg
                
        if self.show_console_print == True:
            print("Segments:" , segments)
            print("spectrum components: " , nr_spectrum)
            print("Size of the feature vector: ", np.size(feature_vector))
            print("The feature vector: ", feature_vector)

        return feature_vector

