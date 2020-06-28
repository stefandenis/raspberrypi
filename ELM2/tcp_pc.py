from scipy.io import wavfile
import socket 
import pyaudio
import time
import struct 
import wave 
import numpy as np

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_address = ('192.168.0.108',60000)
print('starting up on %s port %s' %(server_address, str(10000)))

sock.bind(server_address)

sock.listen(4)




while True:
	print("waiting for connection")
	connection, client_address = sock.accept()
    
	print("connection from: ", client_address)
	try:
		while True:
			data = connection.recv(2048*2*2)
         
        
          
			#print('received from client: ',data))
			if data:
				print("received message: %s" % data)
				connection.sendall(bytes('OK','utf-8'))
			else:
				print("no more data from: ", client_address)
				break

            
    
	except KeyboardInterrupt:
		connection.close()
		break
    
