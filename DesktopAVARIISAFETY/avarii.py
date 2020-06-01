from time import sleep

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)

# avarii LEDS init
GPIO.setup(40, GPIO.OUT)
GPIO.output(40,False) 

while(True):
 	
	GPIO.output(40,True)
	sleep(0.5)
	GPIO.output(40,False)
	sleep(0.5)


print("avarii ended")

GPIO.cleanup()