from time import sleep

import RPi.GPIO as GPIO
import os
import socket
from scipy.io import wavfile
import time
import wave
sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 

server_address2 = ('192.168.0.196',40000)

avarii_flag = False 
lumini_flag = False
def init():
	global pwm_lw
	global pwm_rw	
	GPIO.setmode(GPIO.BOARD)
	PWM_lw_pin = 12
	PWM_rw_pin = 13
	GPIO.setup(PWM_lw_pin,GPIO.OUT)
	GPIO.setup(PWM_rw_pin,GPIO.OUT)
	GPIO.setup(16,GPIO.OUT)
	GPIO.setup(18,GPIO.OUT)
	GPIO.setup(11,GPIO.OUT)
	GPIO.setup(15,GPIO.OUT)
	pwm_lw = GPIO.PWM(PWM_lw_pin,100)
	pwm_rw = GPIO.PWM(PWM_rw_pin,100)
	pwm_lw.start(0)
	pwm_rw.start(0)
	
	#LED FRANA
	GPIO.setup(38,GPIO.OUT)

	#SAFETY CHECK
	GPIO.setup(35,GPIO.IN)
	
	#LED LUMINI
	GPIO.setup(22,GPIO.OUT)


def forward(duty):
	pwm_lw.ChangeDutyCycle(duty)
	# left wheel forward	
	GPIO.output(16,True)
	GPIO.output(18,False)

	pwm_rw.ChangeDutyCycle(duty)
	# right wheel forward
	GPIO.output(11,False)
	GPIO.output(15,True)

def backward(duty):
	pwm_lw.ChangeDutyCycle(duty)
	#left wheel backward
	GPIO.output(16,False)
	GPIO.output(18,True)

	pwm_rw.ChangeDutyCycle(duty)
	#right wheel backward
	GPIO.output(11,True)
	GPIO.output(15,False)


def start_lw(duty):
	pwm_lw.start(duty)

def start_rw(duty):
	pwm_rw.start(duty)

def stop_lw():
	pwm_lw.ChangeDutyCycle(0)

def stop_rw():
	pwm_rw.ChangeDutyCycle(0)

def stop():
        pwm_rw.ChangeDutyCycle(0)
        pwm_lw.ChangeDutyCycle(0)
    
def start(duty):
        pwm_lw.start(duty)
        pwm_rw.start(duty)
    
def avarii():
	global avarii_flag
	avarii_flag = not avarii_flag
	if avarii_flag: 
		os.system("python3 /home/pi/Desktop/avarii.py &") 	
	else:
		os.system("pkill -f avarii.py")
		GPIO.setup(40,GPIO.OUT)
		GPIO.output(40,False)
	
def frana(duty):
	GPIO.output(38,True)
	for i in range(duty,0,-1): 
		sleep(0.01)
		pwm_rw.ChangeDutyCycle(i)
		pwm_lw.ChangeDutyCycle(i)

	GPIO.output(38,False)
	

def lumini():
	global lumini_flag
	lumini_flag = not lumini_flag
	
	GPIO.output(22,lumini_flag)

def hey_jarvis():
	BUFFER_SIZE = 1024	
	global sock
	global server_address2
	wf = wave.open('/home/pi/Desktop/jarvis.wav','rb')
	data = wf.readframes(BUFFER_SIZE)

	sent = sock.sendto(data,server_address2)
	while data != b'':
		sent = sock.sendto(data,server_address2)    
		response,addr = sock.recvfrom(1024)
		data = wf.readframes(BUFFER_SIZE)


def radio():
	BUFFER_SIZE = 1024	
	global sock
	global server_address2
	wf = wave.open('/home/pi/Desktop/radio.wav','rb')	
	size = 0	
	data = wf.readframes(BUFFER_SIZE)
	size = size + BUFFER_SIZE
	sent = sock.sendto(data,server_address2)
	while data != b'':
		sent = sock.sendto(data,server_address2)    
		response,addr = sock.recvfrom(1024)
		data = wf.readframes(BUFFER_SIZE)
		size = size + BUFFER_SIZE
		if size > 1600000:
			break


