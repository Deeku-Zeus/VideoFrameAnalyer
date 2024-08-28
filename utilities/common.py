import hashlib
import time
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_unique_hash():
    # Combine a unique identifier with current time and random bytes
    unique_string = f"{time.time()}-{os.urandom(16)}"
    
    # Create a SHA-256 hash
    hash_object = hashlib.sha256(unique_string.encode('utf-8'))
    unique_hash = hash_object.hexdigest()
    
    return unique_hash

def calculate_iou(box1, box2):
    # Box format: [x1, y1, x2, y2]
    
    # Calculate intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x1_inter < x2_inter and y1_inter < y2_inter:
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    else:
        intersection_area = 0
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def is_within(outer_box, inner_box):
    # Check if inner_box is within outer_box
    return (outer_box[0] <= inner_box[0] and
            outer_box[1] <= inner_box[1] and
            outer_box[2] >= inner_box[2] and
            outer_box[3] >= inner_box[3])

def filter_overlapping_entries(data, iou_threshold=0.5):
    keep_indices = set()
    remove_indices = set()

    for i, item in enumerate(data):
        if i in remove_indices:
            continue
        
        current_box = item['dress']['coordinates']
        current_confidence = item['dress']['confidence']
        to_keep = True
        
        for j, other_item in enumerate(data):
            if i == j or j in remove_indices:
                continue
            
            other_box = other_item['dress']['coordinates']
            other_confidence = other_item['dress']['confidence']
            
            if is_within(other_box, current_box):
                if current_confidence < other_confidence:
                    to_keep = False
                    remove_indices.add(i)
                    break
                else:
                    remove_indices.add(j)
            elif is_within(current_box, other_box) and calculate_iou(current_box, other_box) > iou_threshold:
                if current_confidence < other_confidence:
                    to_keep = False
                    remove_indices.add(i)
                    break
                else:
                    remove_indices.add(j)
        
        if to_keep:
            keep_indices.add(i)
    
    # Construct the filtered data based on the indices to keep
    filtered_data = [data[i] for i in keep_indices]
    
    return filtered_data

def save_image_with_detections(image_path, output_path, detections):
    """
    Save the image with bounding boxes for detected objects.

    Parameters:
    - image_path (str): Path to the original image file.
    - output_path (str): Path to save the image with detections.
    - detections (str): JSON string containing detected objects with labels and coordinates.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Plot the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    # Load detection data from JSON
    detection_data = json.loads(detections)

    # Draw bounding boxes and labels on the image
    for detection in detection_data:
        id, details = next(iter(detection.items()))  # Get the label and details
        coordinates = details["coordinates"]
        score = details["confidence"]
        uid = details["uid"]
        label_text = details["tags"]

        xmin, ymin, xmax, ymax = coordinates
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        plt.text(xmin, ymin, f"{label_text}: {score:.2f} ({uid[:6]})", color="white", fontsize=12, backgroundcolor="red")

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print("File Saved !!")