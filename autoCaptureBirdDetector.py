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

# ===== GPIO Setup =====
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# ===== Paths =====
HOME_DIR = os.environ['HOME']
SAVE_DIR = f"{HOME_DIR}/Desktop/field_test_images"
os.makedirs(SAVE_DIR, exist_ok=True)

#File Paths need to be updated to point to the model's location on computer that's running this script
YOLO_MODEL_PATH = "/path/to/YOLOdetector/directionary/best.pt"
GOOSE_MODEL_PATH = "/path/to/binaryClassifier/dictionary/isGoose_model.pth"

LOG_FILE = f"{SAVE_DIR}/field_log.csv"

# Create CSV header if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("timestamp,image_file,bird_detected,bird_confidence,goose_detected,goose_confidence,goose_label\n")

# ===== Load YOLO =====
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

# ===== Load Goose Classifier =====
print("Loading Goose classifier...")
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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ===== Camera Setup =====
print("Initializing camera...")
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
time.sleep(2)  # Allow camera to warm up
print("Camera ready.\n")


# ===== Capture Image =====
def capture_image():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"image_{timestamp}.jpg"
    image_path = f"{SAVE_DIR}/{image_filename}"

    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, frame_bgr)

    print(f"[{timestamp}] Image captured: {image_filename}")
    return image_path, image_filename, timestamp


# ===== YOLO Bird Detection =====
def isBird(image_path):
    results = yolo_model(image_path, conf=0.1)

    bird_detected = False
    bird_conf = 0.0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if cls_id == 0:  # class 0 = bird
                bird_detected = True
                bird_conf = max(bird_conf, conf)

    return bird_detected, bird_conf


# ===== Goose Classifier =====
def isGoose(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = goose_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = goose_model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = goose_classes[predicted.item()]
    conf = confidence.item()
    goose_detected = (label == "goose")

    return goose_detected, conf, label


# ===== Log Result =====
def log_result(timestamp, image_filename, bird, bird_conf, goose, goose_conf, goose_label):
    with open(LOG_FILE, "a") as f:
        f.write(
            f"{timestamp},{image_filename},{bird},{float(bird_conf):.4f},"
            f"{goose},{float(goose_conf):.4f},{goose_label}\n"
        )


# ===== Print Result =====
def print_result(timestamp, image_filename, bird, bird_conf, goose, goose_conf, goose_label):
    print("=" * 50)
    print(f"  Timestamp   : {timestamp}")
    print(f"  Image File  : {image_filename}")
    print(f"  --- YOLO Bird Detection ---")
    print(f"  Bird Detected  : {bird}")
    print(f"  Bird Confidence: {float(bird_conf):.4f}")
    print(f"  --- Goose Classifier ---")
    print(f"  Goose Detected  : {goose}")
    print(f"  Goose Confidence: {float(goose_conf):.4f}")
    print(f"  Goose Label     : {goose_label}")
    print("=" * 50 + "\n")


# ===== MAIN LOOP =====
CAPTURE_INTERVAL = 3  # seconds between captures

print("Auto-capture started. Press Ctrl+C to stop.\n")

try:
    while True:
        loop_start = time.time()

        # 1. Capture
        image_path, image_filename, timestamp = capture_image()

        # 2. YOLO bird detection
        bird, bird_conf = isBird(image_path)

        # 3. Goose classification (only if bird found)
        goose = False
        goose_conf = 0.0
        goose_label = "N/A"

        if bird:
            goose, goose_conf, goose_label = isGoose(image_path)

        # 4. Print results
        print_result(timestamp, image_filename, bird, bird_conf, goose, goose_conf, goose_label)

        # 5. Save results to CSV
        log_result(timestamp, image_filename, bird, bird_conf, goose, goose_conf, goose_label)

        # 6. Wait for the remainder of the 3-second interval
        elapsed = time.time() - loop_start
        sleep_time = max(0, CAPTURE_INTERVAL - elapsed)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nStopping auto-capture...")

finally:
    picam2.stop()
    GPIO.cleanup()
    print("Camera stopped. GPIO cleaned up. Exiting.")
