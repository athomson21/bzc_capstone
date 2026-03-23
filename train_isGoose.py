import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from PIL import Image
import cv2


# ------------------
# Settings
# ------------------
#data_dir = r"C:\Users\athom\OneDrive\Python\isGooseDataset"
data_dir = r"C:\Users\athom\OneDrive\Desktop\zoo_bird_detection\isGooseV3"

train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "valid")

batch_size = 16
num_epochs = 30
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------
# Transforms
# ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------
# Dataset
# ------------------
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

print(train_dataset.class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ------------------
# Model
# ------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# ------------------
# Loss + Optimizer
# ------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

class_weights = torch.tensor([1.3, 1.0]).to(device) #higher weight for goose class
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ------------------
# Training Loop
# ------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    scheduler.step()

#Validation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total
print(f"Validation Accuracy: {val_accuracy:.2f}%")


# ------------------
# Save Model
# ------------------
torch.save(model.state_dict(), "isGoose_model.pth")
print("Model saved as isGoose_model.pth")
