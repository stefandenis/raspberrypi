import RPi.GPIO as GPIO
import time as ti
import sys
import os


sys.path.append('/home/pi/Desktop/licenta/licenta2/ELM2')

#from driver import *

GPIO.setmode(GPIO.BOARD)


TRIG = 36
ECHO = 32

GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)

#GPIO.setup(37,GPIO.OUT)

def stop():
	GPIO.output(37,True)
	ti.sleep(0.2)
	GPIO.output(37,False)
	
def distance():
	GPIO.output(TRIG,True)
	ti.sleep(0.00001)
	GPIO.output(TRIG,False)	

	pulse_start = ti.time()
	pulse_end = ti.time()

	while GPIO.input(ECHO) == 0:
		pulse_start = ti.time()

	while GPIO.input(ECHO) == 1:
		pulse_end = ti.time()

	time = pulse_end - pulse_start

	distance = (time *34300)/2

	return distance 


while True:
	dist = distance()
	#print(dist)
	if dist < 5:
		stop()
	ti.sleep(0.1)
