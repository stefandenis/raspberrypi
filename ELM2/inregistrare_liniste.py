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
chunk = 4096 # 2^12 samples for buffer
record_secs = 2 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii)

os.chdir('liniste')

for i in range(200):
    print(i)
    wav_output_filename = 'liniste0' + str(i) + '.wav'

    audio = pyaudio.PyAudio() # create pyaudio instantiation
    sleep(0.5)



    frames = []
    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                        input_device_index = dev_index,input = True, \
                        frames_per_buffer=chunk)
    print("recording")

    # loop through stream and append audio chunks to frame array

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

