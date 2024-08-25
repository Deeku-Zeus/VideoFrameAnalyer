# model/yolo_detection.py

from ultralytics import YOLO
from PIL import Image
import numpy as np

def load_yolo_model():
    # Load the pre-trained YOLOv8 model (e.g., YOLOv8s)
    model = YOLO("yolov8s.pt")  # Ensure this path is correct or use a valid URL
    return model

def detect_objects(image_path, max_objects=5):
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

def get_detect_objects_list(image, max_objects=5):
    # Load the YOLOv8 model
    model = load_yolo_model()
    class_names = model.names
    # Load image
    img = image

    # Perform detection
    results = model(img)

    # Extract results
    detections = results[0].boxes.xyxy.numpy()  # Access bounding boxes (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.numpy()  # Access confidences
    class_ids = results[0].boxes.cls.numpy()     # Access class IDs

    # Create a list of detections with their confidence scores
    detections_with_confidence = [
        (box, conf,class_id) for box, conf, class_id in zip(detections, confidences,class_ids)
    ]

    # Sort detections by confidence in descending order
    detections_with_confidence.sort(key=lambda x: x[1], reverse=True)

    # Select the top N detections (e.g., max_objects)
    selected_detections = detections_with_confidence[:max_objects]

    # Process selected detections
    object_list = []
    for (box, conf,class_id) in selected_detections:
        x1, y1, x2, y2 = box
        object_list.append({class_names[class_id]:{'coordinates':[int(x1), int(y1), int(x2 - x1), int(y2 - y1)], 'confidence':str(conf)}})  # [x, y, width, height]

    return object_list
