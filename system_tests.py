import RPi.GPIO as GPIO
import time
from time import sleep
from datetime import datetime
#from picamzero import Camera
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
from picamera2 import Picamera2

#GPIO SETUP
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

trig = 21
echo = 20
led = 18
# solenoid = 4
# motor = 1
button = 17
direction = 27
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

# _____ Paths ______
HOME_DIR = os.environ['HOME']
SAVE_DIR=f"{HOME_DIR}/Desktop/field_test_images"
os.makedirs(SAVE_DIR, exist_ok=True)

YOLO_MODEL_PATH = "/home/bzccapstone/Desktop/modelVersions/best.pt"
GOOSE_MODEL_PATH = "/home/bzccapstone/Desktop/modelVersions/GooseModels/isGoose_model.pth"

LOG_FILE = f"{SAVE_DIR}/field_log.csv"
#create CSV header if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp, bird_detected, bird_confidence, goose_detected, goose_confidence, goose_label\n")

# ____ Load Yolo ____
yolo_model = YOLO(YOLO_MODEL_PATH)

# ____ Load Goose Classifier ____
device = torch.device("cpu")

goose_model = models.resnet18(weights=None)
goose_model.fc = nn.Linear(goose_model.fc.in_features, 2)
goose_model.load_state_dict(torch.load(GOOSE_MODEL_PATH, map_location=device))
goose_model.to(device)
goose_model.eval()

goose_classes = ["goose", "not goose"]

goose_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#picam2 = Picamera2()
#picam2.configure(picam2.create_still_configuration())

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

# ____ Camera Function ____ *with preview 
def capture_image():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"{SAVE_DIR}/image_{timestamp}.jpg"
    
    picam2 = None
    for attempt in range(5):
        try:
            os.system("fuser -k /dev/video0")
            time.sleep(3)
            picam2 = Picamera2()
            break  #success, exit retry loop
        except Exception as e:
            print(f"Camera init attempt {attempt+1} failed: {e}")
            time.sleep(3)
#    picam2.configure(picam2.create_still_configuration())
#    picam2.start()
    try:
        picam2.configure(picam2.create_still_configuration())
        picam2.start()
        time.sleep(0.5)
    
        frame = picam2.capture_array()
    
        #Convert color for OpenCv
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(image_path, frame_bgr)
        print(f"Image captured: {image_path}")
    
        #Resizze just for preview (saved image stays full res)
        preview = cv2.resize(frame_bgr, (820, 616))
        screen_w, screen_h = 1920, 1080
        x = (screen_w - 820) // 2
        y = (screen_h - 616) // 2
        cv2.imshow("Preview", preview)
        cv2.moveWindow("Preview", x, y)
        cv2.waitKey(2000) #show for 2 seconfds
        cv2.destroyAllWindows()


    finally:
        picam2.stop()
        picam2.close()
        time.sleep(3)
        
    GPIO.output(led, False)
    return image_path, timestamp

    
#______ YOLO Bird Check _____
def isBird(image_path):
    results = yolo_model(image_path, conf=0.1)
    
    bird_detected = False
    bird_conf = 0.0
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            #label = yolo_model.names[cls_id]
            conf = float(box.conf)
            
            if cls_id == 0:	#class 0 = bird 
                bird_detected = True
                bird_conf = max(bird_conf, conf) #keeps highest confidence detection
                break
            
    return bird_detected, bird_conf


# _____ Goose Check _____
def isGoose(image_path):
    image = Image.open(image_path).convert("RGB")
    image = goose_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = goose_model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    
    label = goose_classes[predicted.item()]
    conf = confidence.item()
    
    goose_detected = (label == "goose") and (conf > 0.6)
    
    print(f"Goose classifier: {label} ({conf:.2f})")    
    return goose_detected, conf, label
    
# def cameraOn():
#     home_dir = os.environ['HOME']
#     cam = Camera()
#     cam.start_preview()
#     cam.take_photo(f"{home_dir}/Desktop/image.jpg")
#     cam.stop_preview()
#     image_path = f"{home_dir}/Desktop/image.jpg"
#     image = Image.open(image_path)
#     time.sleep(5)
#     print("turning on camera")
#     return image


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

# _____ Logging Function ____
        
def log_result(timestamp, bird, bird_conf, goose, goose_conf, goose_label):
    with open(LOG_FILE, "a") as f:
        bird_conf = float(bird_conf)
        goose_conf = float(goose_conf)
        f.write(f"{timestamp}, {bird}, {bird_conf:.3f}, {goose}, {goose_conf:.3f}, {goose_label}\n")

# _____ MAIN LOOP _____
print("System ready. Press button to capture image.")

# def main():
#     while True:
#         #if GPIO.input(BUTTON_PIN) == GPIO.LOW:   #LOW = button pressed (pull-up config)
#         print("\nButton Pressed")
#         image_path, timestamp = capture_image()
#         bird, bird_conf = isBird(image_path)
#             
#         goose = False
#         goose_conf = 0.0
#         goose_label = "N/A"
#             
#         if bird:
#             goose, goose_conf, goose_label = isGoose(image_path)
#                 
#         #Console Output (clean ad minimal)
#         print("\n --- Predition ---")
#         print(f"Bird Detected: {bird} | Confidence: {bird_conf:.2f}")
#         print(f"Goose Detected: {goose} | Confidence: {goose_conf:.2f}")
#         print("---------------------\n")
#             
#         log_result(timestamp, bird, bird_conf, goose, goose_conf, goose_label)
#             
#         #Wait for button to be released before listening again
#         while GPIO.input(BUTTON_PIN) == GPIO.LOW:
#             time.sleep(0.05)
#         time.sleep(0.1)
#             
# #            time.sleep(2)
#     time.sleep(0.1)
      
def main():
    while True:
        print("\n\nCamera off")
        dist, motion = distance()
        #print("Measured Distance = %.1f cm" % dist)
        print("Measured Distance = %.1f cm" % dist)
        time.sleep(1)
        if motion == True:
            print("Turn on camera")
            image_path, timestamp = capture_image()
            bird, bird_conf = isBird(image_path)
            
            goose = False
            goose_conf = 0.0
            goose_label = "N/A"
            
            if bird:
                goose, goose_conf, goose_label = isGoose(image_path)   
            #Console Output (clean ad minimal)
            print("\n --- Predition ---")
            print(f"Bird Detected: {bird} | Confidence: {bird_conf:.2f}")
            print(f"Goose Detected: {goose} | Confidence: {goose_conf:.2f}")
            print("---------------------\n")
            
            if bird and not goose:
                dispense_food()
            
#         log_result(timestamp, bird, bird_conf, goose, goose_conf, goose_label)
#             if bird:
#                 print("Bird Detected - checking goose...")
#                 goose = isGoose(f"{os.environ['HOME']}/Desktop/image.jpg")
#                 
#                 if goose:
#                     label = "Goose"
#                     print("Goose detected - NOT dispensing food")
# #                     dispense_food #putting this here for testing
#                 else:
#                     label = "Non-Goose Bird"
#                     print("Non-goose bird detected - dispensing food.")
#                     dispense_food
#             else:
#                 label = "No Bird"
#                 print("No Bird Detected")
#             
#             log_detection(image, label)
            idle_motor()
            motion = False
        else:
            idle_motion()
        




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
'''GOOSE_MODEL_PATH = "/home/bzccapstone/isGoose_model.pth"
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
'''

        
main() 
