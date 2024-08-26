import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO  # Import YOLO from ultralytics package
import json

def load_models():
    """
    Load YOLOS Fashionpedia and YOLOv8 models.
    """
    # Load YOLOS Fashionpedia model
    processor_fashion = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
    model_fashion = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
    
    # Load YOLOv8 model (e.g., 'yolov8n' for nano, 'yolov8s' for small, etc.)
    model_coco = YOLO('yolov8n.pt')  # Using the nano model for faster inference
    
    return processor_fashion, model_fashion, model_coco

def detect_objects_fashion(image_path, processor, model, max_objects=5):
    """
    Detect fashion objects using YOLOS Fashionpedia model.
    """
    selected_classes_fashion = [
        "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "jacket", 
        "vest", "pants", "shorts", "skirt", "coat", "dress", 
        "glasses", "hat", "headband, head covering, hair accessory", 
        "tie", "glove", "watch", "belt", "leg warmer", "tights, stockings", 
        "sock", "shoe", "bag, wallet", "scarf", "umbrella", "hood"
    ]

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score > 0.5:  # Confidence threshold
            label_text = model.config.id2label[label.item()]
            if label_text in selected_classes_fashion:
                box = box.tolist()
                detections.append({
                    "label": label_text,
                    "score": round(score.item(), 2),
                    "box": box
                })
                if len(detections) >= max_objects:
                    break
    return detections

def detect_objects_coco(image_path, model, selected_classes_coco, max_objects=5):
    """
    Detect objects using a YOLOv8 model trained on COCO.
    """
    results = model(image_path)
    detections = []
    for result in results:
        for box in result.boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            if cls in selected_classes_coco and conf > 0.5:
                detections.append({
                    "label": model.names[cls],
                    "score": round(conf, 2),
                    "box": [xmin, ymin, xmax, ymax]
                })
                if len(detections) >= max_objects:
                    break
    return detections

def combine_and_save_detections(image_path, output_path, detections_fashion, detections_coco):
    """
    Combine detections from YOLOS Fashionpedia and YOLO COCO, and save the image with bounding boxes.
    """
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    all_detections = detections_fashion + detections_coco

    for detection in all_detections:
        box = detection["box"]
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        plt.text(xmin, ymin, f"{detection['label']}: {detection['score']:.2f}", color="white", fontsize=12, backgroundcolor="red")

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

# Example usage:
if __name__ == "__main__":
    image_path = os.path.expanduser("~/Downloads/test0.jpeg")
    output_path = os.path.expanduser("~/Downloads/test0_combined_detections.png")
    
    processor_fashion, model_fashion, model_coco = load_models()
    
    selected_classes_coco = [0, 1, 2, 24, 26]  # Person, Bicycle, Car, Backpack, Handbag
    
    detections_fashion = detect_objects_fashion(image_path, processor_fashion, model_fashion, max_objects=5)
    detections_coco = detect_objects_coco(image_path, model_coco, selected_classes_coco, max_objects=5)
    
    combine_and_save_detections(image_path, output_path, detections_fashion, detections_coco)
    
    print("Combined detections saved.")