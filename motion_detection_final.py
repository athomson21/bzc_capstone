import RPi.GPIO as GPIO
import time
from picamera import Camera
import osGPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

trig = 21
echo = 20
led = 18
solenoid = 35

GPIO.setup(led, GPIO.OUT)
GPIO.setup(trig, GPIO.OUT)
GPIO.setup(echo, GPIO.IN)
GPIO.setup(solenoid, GPIO.OUT)

def distance():
    GPIO.output(trig, True)
    print("sending pulse")
    time.sleep(0.00001)
    GPIO.output(trig, False)
    print("pulse done")

    StartTime = time.time()
    StopTime = time.time()

    #saving start time
    while GPIO.input(echo) == 0:
        StartTime = time.time()
    while GPIO.input(echo) == 1:
        StopTime = time.time()
        GPIO.output(led, True)
    TimeElapsed = StopTime - StartTime
    #multiply to get distance
    distance = (TimeElapsed * 34300) / 2

    if distance < 100:
        print("Motion detected")
        motion = True
    else:
        motion = False
    return distance, motion

def camera_on():
    home_dir = os.environ['HOME']
    cam = Camera()
    cam.start.preview()
    cam.take_photo(f"{home_dir}/Desktop/image.jpg")
    cam.stop_preview()
    image_path = f"{home_dir}/Desktop/image.jpg"
    image = Image.open(image_path)
    time.sleep(5)
    print("turning on camera")
    return image

def solenoid_active():
    dispense_done = False
    time.sleep(3)
    GPIO.output(solenoid, True) #solenoid lowered
    time.sleep(3)
    GPIO.output(solenoid, False) #solenoid raised
    dispense_done = True
    print("\nDispense done")
    return dispense_done

def main():
    while True:
        print("Camera off")
        dist, motion = distance()
        print("Measured Distance = %.1fcm" % dist)
        time.sleep(1)
        GPIO.output(solenoid, GPIO.LOW)
        GPIO.output(solenoid, GPIO.HIGH)
        GPIO.output(solenoid, GPIO.LOW)
        print("Camera off")
        solenoid_active()
        while motion == True:
            print("Turn on camera")
            image = camera_on()
            #is_bird(image)
            #is_goose(image)
            time.sleep(1)
            motion = False
main()
