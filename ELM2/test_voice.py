import pyaudio
import wave
import sys
import math
import matplotlib.pyplot as plt
BUFFER_SIZE = 1
#OUTPUT FILE OPTIONS 
WAVE_OUTPUT_FILENAME = "avarii2.wav"
# Opening audio file as binary data
wf = wave.open(r'/home/pi/Desktop/jarvis.wav', 'rb')
# Instantiate PyAudio
p = pyaudio.PyAudio()
file_sw = wf.getsampwidth()
print(file_sw)
print("channels: " ,wf.getnchannels())
print("sampwidth: ",wf.getsampwidth())
print("framerate: " ,wf.getframerate())
print("p.get_format_from_width(file_sw): ", p.get_format_from_width(file_sw))
stream = p.open(format=p.get_format_from_width(file_sw),
                channels=2,
                rate=48000,
                output=True
                #stream_callback = callback
                )
data = wf.readframes(BUFFER_SIZE)
frames = [] 
int_frames = []
while data != b'':
    stream.write(data)
    #print(data)
    #print(data)
    frames.append(data)
    #print("length of data: ", len(data))
    #print("one byte")
    data = wf.readframes(BUFFER_SIZE)
    #print(data[2])
stream.stop_stream()
stream.close()
p.terminate()