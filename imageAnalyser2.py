import torch
import cv2
import os
import matplotlib.pyplot as plt

# Load YOLOv5 model (small version)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load an image
img = os.path.join("E:\\SystemFiles", "Pictures", "nanna trip.jpg")

# Perform inference
results = model(img)

# Print detected labels and confidence scores
for result in results.xyxy[0]:
    label = model.names[int(result[5])]  # Get class name
    confidence = result[4].item()  # Get confidence score
    print(f"Detected: {label} with confidence: {confidence:.2f}")

# Show the image with bounding boxes
results.show()