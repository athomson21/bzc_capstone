import RPi.GPIO as GPIO
import time
import os
from datetime import datetime
from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ___ GPIO Setup _____
BUTTON_PIN = 13
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

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

goose_classes = ["non-goose", "goose"]

goose_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())

# ____ Camera Function ____ *with preview 
def capture_image():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"{SAVE_DIR}/image_{timestamp}.jpg"
    
#    picam2 = Picamera2()
#    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    
    frame = picam2.capture_array()
    
    #Convert color for OpenCv
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    #Resizze just for preview (saved image stays full res)
    preview = cv2.resize(frame_bgr, (820, 616))
    screen_w, screen_h = 1920, 1080
    x = (screen_w - 820) // 2
    y = (screen_h - 616) // 2
    
#    cv2.imwrite(image_path, frame)
    
    cv2.imshow("Preview", preview)
    cv2.moveWindow("Preview", x, y)
    cv2.waitKey(2000) #show for 2 seconfds
    cv2.destroyAllWindows()
    
#    picam2.start_preview(Preview.QTGL)
#    time.sleep(2)
    
#    picam2.capture_file(image_path)
#    picam2.stop_preview()
    cv2.imwrite(image_path, frame_bgr)
    picam2.stop()
    
    print(f"Image captured: {image_path}")
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
    
    goose_detected = (label == "goose")
    
    return goose_detected, conf, label


# _____ Logging Function ____
        
def log_result(timestamp, bird, bird_conf, goose, goose_conf, goose_label):
    with open(LOG_FILE, "a") as f:
        bird_conf = float(bird_conf)
        goose_conf = float(goose_conf)
        f.write(f"{timestamp}, {bird}, {bird_conf:.3f}, {goose}, {goose_conf:.3f}, {goose_label}\n")

# _____ MAIN LOOP _____
print("System ready. Press button to capture image.")

def main():
    while True:
        #if GPIO.input(BUTTON_PIN) == GPIO.LOW:   #LOW = button pressed (pull-up config)
        print("\nButton Pressed")
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
            
        log_result(timestamp, bird, bird_conf, goose, goose_conf, goose_label)
            
        #Wait for button to be released before listening again
        while GPIO.input(BUTTON_PIN) == GPIO.LOW:
            time.sleep(0.05)
        time.sleep(0.1)
            
#            time.sleep(2)
    time.sleep(0.1)
                
'''
image = Image.open(image_path).convert("RGB")
image = goose_transform(image).unsqueeze(0).to(device)
with torch.no_grad():
    outputs = goose_model(image)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)
                    
goose_label = goose_classes[predicted.item()]
goose_conf = confidence.item()
                
                if goose:
                    decision = "block"
                else:
                    decision = "allow"
            else:
                decision = "no_bird"
                
            
            log_result(timestamp, bird, goose_label, goose_conf, decision)
        
# except KeyboardInterrupt:
#     print("shutting down...")
#     GPIO.cleanup()'''
main()