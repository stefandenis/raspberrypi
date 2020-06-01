import pyaudio
import wave
import numpy as np
import scipy.io as sio
import scipy.io.wavfile
from time import sleep
import os
form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 8192 # 2^12 samples for buffer
record_secs = 2 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii)

print(os.getcwd())
os.chdir('../commands_test/radio_test')


for i in range(10):
    audio = pyaudio.PyAudio() # create pyaudio instantiation
    sleep(0.5)
    print(i)
    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)


    wav_output_filename = 'radio_test0' + str(i) + '.wav'
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
    frames = []
