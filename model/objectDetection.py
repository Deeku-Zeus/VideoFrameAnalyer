# model/yolo_detection.py

import torch
import multiprocessing
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set the number of threads for PyTorch to use
torch.set_num_threads(multiprocessing.cpu_count())

def load_yolo_model():
    # Load the pre-trained YOLOv8 model (e.g., YOLOv8s)
    model = YOLO("yolov8s.pt")  # Ensure this path is correct or use a valid URL
    return model

def detect_objects(image_path, image_name, max_objects=5):
    # Load the YOLOv8 model
    model = load_yolo_model()

    # Load image
    img = Image.open(image_path)

    # Perform detection
    results = model(img)

    # Extract results
    detections = results[0].boxes.xyxy.numpy()  # Access bounding boxes (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.numpy()  # Access confidences
    class_ids = results[0].boxes.cls.numpy()     # Access class IDs

    # Create a list of detections with their confidence scores
    detections_with_confidence = [
        (box, conf) for box, conf in zip(detections, confidences)
    ]

    # Sort detections by confidence in descending order
    detections_with_confidence.sort(key=lambda x: x[1], reverse=True)

    # Select the top N detections (e.g., max_objects)
    selected_detections = detections_with_confidence[:max_objects]

    # Process selected detections
    boxes = []
    for (box, conf) in selected_detections:
        x1, y1, x2, y2 = box
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])  # [x, y, width, height]

    return boxes

def get_detect_objects_list(image, image_name, max_objects=5):
    # Load the YOLOv8 model
    model = load_yolo_model()
    class_names = model.names
    print(class_names)
    selected_classes = ['bicycle', 'car', 'motorcycle', 'truck', 
                'boat','bench', 'backpack', 'umbrella', 'handbag', 
               'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 
               'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 
               'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 
               'teddy bear', 'hair drier', 'toothbrush']

    # Load image
    img = image

    # Perform detection
    results = model(img)

    # Extract results
    detections = results[0].boxes.xyxy.numpy()  # Access bounding boxes (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.numpy()  # Access confidences
    class_ids = results[0].boxes.cls.numpy()     # Access class IDs

    # Create a list of detections with their confidence scores and filter by selected classes
    detections_with_confidence = [
        (box, conf, class_id) 
        for box, conf, class_id in zip(detections, confidences, class_ids) 
        if class_names[int(class_id)] in selected_classes
    ]

    # Sort detections by confidence in descending order
    detections_with_confidence.sort(key=lambda x: x[1], reverse=True)

    # Select the top N detections (e.g., max_objects)
    selected_detections = detections_with_confidence[:max_objects]

    # Process selected detections
    object_list = []
    for (box, conf, class_id) in selected_detections:
        x1, y1, x2, y2 = box
        object_list.append({
            class_names[int(class_id)]: {
                'coordinates': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],  # [x, y, width, height]
                'confidence': str(conf),
                'crop_image_name': image_name[:-4]
            }
        })

    return object_list
