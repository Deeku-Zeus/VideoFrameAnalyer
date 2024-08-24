import os
import torch
from torchvision import models, transforms
from PIL import Image

# Load pre-trained YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load pre-trained ResNet for classification
classification_model = models.resnet50(pretrained=True)
classification_model.eval()

# Preprocessing pipeline for classification
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image for testing
#img = 'path_to_image.jpg'
#img = os.path.join("C:\\Users", "deeku", "OneDrive", "Pictures", "sample.jpg")
#"E:\SystemFiles\Pictures\nanna trip.jpg"
img = os.path.join("E:\\SystemFiles", "Pictures", "nanna trip.jpg")

# Detect objects
results = yolo_model(img)

# Extract detected objects
for obj in results.xyxy[0]:  # xyxy: coordinates of the bounding boxes
    x1, y1, x2, y2 = map(int, obj[:4])
    label = obj[5].item()

    # Crop the object from the image
    image = Image.open(img)
    cropped_img = image.crop((x1, y1, x2, y2))

    # Classify the object type
    input_tensor = preprocess(cropped_img).unsqueeze(0)
    with torch.no_grad():
        output = classification_model(input_tensor)
    _, predicted_class = output.max(1)

    print(f"Detected {label} with classification: {predicted_class.item()}")
