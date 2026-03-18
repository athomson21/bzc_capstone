import RPi.GPIO as GPIO
import time
from time import sleep
from datetime import datetime
from picamzero import Camera
from PIL import Image
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'usr/lib/arm-linux-gnueabihf/gt5/plugins/platforms' #import this before cv2
from ultralytics import YOLO
import cv2
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
#import sysconfig
import sys
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

trig = 21
echo = 20
led = 18
# solenoid = 4
# motor = 1
button = 17
direction = 16
step = 13
reset = 26
delay = 1
sec = 1
step_delay = 0.01
GPIO.setup(led, GPIO.OUT)
GPIO.setup(trig, GPIO.OUT) #trig
GPIO.setup(echo, GPIO.IN) #echo
GPIO.setup(button, GPIO.IN)
GPIO.setup(direction, GPIO.OUT)
GPIO.setup(step, GPIO.OUT)
GPIO.setup(reset, GPIO.OUT)
#sysconfig.get_path("data") #used with sysconfig to get local environment path
sys.prefix #used to get local environment path with import sys

def distance():
    GPIO.output(trig, True) #pulse
    print("sending pulse")
    time.sleep(0.00001) #0.01ms low
    GPIO.output(trig, False)
    print("pulse done")
    
    StartTime = time.time()
    StopTime = time.time()
    
    #saving startime
    while GPIO.input(echo) == 0:
        StartTime = time.time()
        
    while GPIO.input(echo) == 1:
        StopTime = time.time()
        GPIO.output(led, True) #turn on led
        
    TimeElapsed = StopTime - StartTime
    #multiply to get distance
    distance = (TimeElapsed * 34300) / 2
    
    if distance < 100: #centimeters
        print("Motion detected")
        motion = True
    else:
        motion = False
    
    return distance, motion

def cameraOn():
    home_dir = os.environ['HOME']
    cam = Camera()
    cam.start_preview()
    cam.take_photo(f"{home_dir}/Desktop/image.jpg")
    cam.stop_preview()
    image_path = f"{home_dir}/Desktop/image.jpg"
    image = Image.open(image_path)
    time.sleep(5)
    print("turning on camera")
    return image

def idle_motion():
    '''wait state to save power between ultrasonic sensor readings '''
    time.sleep(10) #our wait state, can change this to whatever we want
def idle_motor():
    ''' wait state to save power between food dispense output'''
    time.sleep(10)
def dispense_food():
    dispense_done = False
    print("Sending reset")
    GPIO.output(direction,True)
    GPIO.output(reset, False)
    sleep(delay)
    GPIO.output(reset, True) #set high
    sleep(delay)
    print("Reset done")
    #osscilate motor
    for i in range(100):
        print("Oscillating motor")
        GPIO.output(step, True)
        sleep(step_delay) #1 us wait
        GPIO.output(step, False)
        sleep(step_delay) # 199ms wait                                                                                              
    print("Ending program")
    sleep(1)
    dispense_done = True
    print("dispense done")
    return dispense_done
      
def main():
    while True:
        print("\n\nCamera off")
        dist, motion = distance()
        #print("Measured Distance = %.1f cm" % dist)
        print("Measured Distance = %.1f cm" % dist)
        time.sleep(1)
        if motion == True:
            print("Turn on camera")
            image = cameraOn() #taking a picture an returning it
            bird = isBird(image)
            if bird:
                print("Bird Detected - checking goose...")
                goose = isGoose(f"{os.environ['HOME']}/Desktop/image.jpg")
                
                if goose:
                    label = "Goose"
                    print("Goose detected - NOT dispensing food")
#                     dispense_food #putting this here for testing
                else:
                    label = "Non-Goose Bird"
                    print("Non-goose bird detected - dispensing food.")
                    dispense_food
            else:
                label = "No Bird"
                print("No Bird Detected")
            
            log_detection(image, label)
            idle_motor()
            motion = False
        else:
            idle_motion()
        
        
# ------ isBird ----- Step 1 of Image Processing        
#path to fine-tuned OLD model 
#MODEL_PATH= "/home/bzccapstone/runs/detect/train7/weights/best.pt"
#Model path to new fine-tuned model (11/20/25) below
MODEL_PATH = "/home/bzccapstone/Desktop/modelVersions/best.pt"

#directory to save detection results
SAVE_DIR = "/home/bzccapstone/Desktop/bird_detections"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_FILE = os.path.join(SAVE_DIR, "detection_log.csv")

#Load YOLO model
model = YOLO(MODEL_PATH)

#--- Bird Detection Helper ---
def isBird(image_path):
    results = model(image_path)
    bird = False
    for r in results:
        for c in r.boxes.cls:
            label = model.names[int(c)]
            if label.lower() == "bird":
                print("Bird detected")
                bird = True
    return bird

def log_detection(image_path, label, log_file=LOG_FILE):

    #ensure directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    #Prepare log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = [timestamp, image_path, label]
    
    #check if file already exists to add headers only once
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "image_path", "label"])
        writer.writerow(log_entry)
        
    print(f"Logged detection: {label} | {image_path}")
       
# ------- Goose Classifier ----
GOOSE_MODEL_PATH = "/home/bzccapstone/isGoose_model.pth"
device = torch.device("cpu")

goose_model = models.resnet18(weights=None)
goose_model.fc=nn.Linear(goose_model.fc.in_features, 2)
goose_model.load_state_dict(torch.load(GOOSE_MODEL_PATH, map_location=device))
goose_model.to(device)
goose_model.eval()

goose_classes = ["non_goose", "goose"]

goose_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def isGoose(image_path):
    image = Image.open(image_path).convert("RGB")
    image = goose_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = goose_model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
    label = goose_classes[predicted.item()]
    conf = confidence.item()
    
    print(f"Goose classifier: {label} ({conf:.2f})")
    
    if label == "goose" and conf > 0.6:
        return True
    return False
        
main() 