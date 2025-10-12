#in terminal run these following codes
'''
sudo apt update
sudo apt install python3-picamzero
'''

from picamzero import Camera #RPI camera
from gpiozero import LED #RPI LED
from time import sleep
import os
import time
import RPi.GPIO as GPIO #RPI GPIO
'''
install Pillow to do image processing
In Linux: pip install Pillow
'''
from PIL import Image

led_output = 23
ultrasonic_pin = 25
motor_output = 14
solenoid_pin = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup(ultrasonic_pin, GPIO.IN)
GPIO.setup(led_output, GPIO.OUT)
GPIO.setup(motor_output, GPIO.OUT)

#functions to be created
def motion_detect():
    if GPIO.input(ultrasonic_pin) == GPIO.HIGH: #input high
        GPIO.output(led_output, GPIO.HIGH) #enable LED
        motion = True #send signal to turn camera on
    else:
        GPIO.output(led_output, GPIO.LOW)
        motion = False
    print("running motion detection")
    return motion

def camera_on():
    home_dir = os.environ['HOME']
    cam = Camera()
    #cam.flip_camera(hflip=True, vflip=True)  # 180 degree rotation
    cam.start_preview()
    cam.take_photo(f"{home_dir}/Desktop/image.jpg") #saves image in that location
    cam.stop_preview()
    image_path = f"{home_dir}/Desktop/image.jpg"
    image = Image.open(image_path)
    sleep(5)  # 5second preview window
    print("turning on camera")
    return image

def is_bird(image):
    #uses OpenCV library to
    contains_bird = 1
    if (contains_bird == 1):
        bird = True
    else:
        bird = False
    print("checking if bird")
    return bird
def is_goose(image):
    contains_goose = 1
    if (contains_goose == 1):
        goose = True
    else:
        goose = False
    print("checking if goose")
    return goose

def dispense_food(bird, goose):
    dispense_done = False
    if bird == True and goose == False:
        #send output signal to solenoid
        GPIO.output(solenoid_pin, GPIO.HIGH)
        sleep(1)
    else:
        GPIO.output(solenoid_pin, GPIO.LOW)
    print("dispensing food")
    return dispense_done
def stop_dispense(dispense_done):
    GPIO.output(solenoid_pin, GPIO.LOW)
    dispense_done = True
    print("stop dispense")
    return dispense_done

def camera_off():
    GPIO.output(led_output, GPIO.LOW)
    #camera already turns off after picture is taken
    #might need to move stuff
    motion = False #signal to stop SM
    print("turning off camera")
    return motion

def main():
    while True:
        motion = motion_detect()
        while motion == True:
            bird = is_bird(camera_on()) #check if valid syntax - should run the function
            if bird == True:
                goose = is_goose(camera_off())
                if goose == False:
                    solenoid = dispense_food(bird, goose)
                    stop_dispense(solenoid)
                    camera_off()
                else:
                    camera_off()
            else:
                camera_off()


main()



