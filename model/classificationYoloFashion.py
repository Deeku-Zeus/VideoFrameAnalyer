import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import sys
sys.path.append('/Users/deekshitswamy/Documents/GitHub/EcomMediaPlayer/VideoFrameAnalyer')
from utilities.common import generate_unique_hash, filter_overlapping_entries

def load_model():
    """
    Load the YOLOS Fashionpedia model and processor.
    """
    processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
    model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")
    return processor, model

def detect_objects(image_path, max_objects=5):
    """
    Detect objects in the image and return a JSON with detected objects and their coordinates.

    Parameters:
    - image_path (str): Path to the image file.
    - max_objects (int): Maximum number of objects to detect. Defaults to 5.

    Returns:
    - result_json (str): JSON string containing detected objects with labels and coordinates.
    """
    # Load the model and processor
    processor, model = load_model()

    # Define the selected class names
    selected_classes = [
        "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "jacket", 
        "vest", "pants", "shorts", "skirt", "coat", "dress", 
        "glasses", "hat", "headband, head covering, hair accessory", 
        "tie", "glove", "watch", "belt", "leg warmer", "tights, stockings", 
        "sock", "shoe", "bag, wallet", "scarf", "umbrella", "hood"
    ]

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Process the image for the model
    inputs = processor(images=image, return_tensors="pt")  # Pass the image object, not the path

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the bounding boxes, labels, and scores
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

    # Prepare the output in JSON format
    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score > 0.5:  # Show only predictions with a high confidence score
            label_text = model.config.id2label[label.item()]
            if label_text in selected_classes:
                box = box.tolist()
                xmin, ymin, xmax, ymax = box
                detections.append({
                    "obj":{
                        "confidence": score.item(),  # Convert to a Python float
                        "coordinates": [xmin, ymin, xmax, ymax],
                        'uid': generate_unique_hash(),
                        'color': 'grey',
                        'tags' : [tag.strip() for tag in label_text.split(",")]
                    }
                })
                if len(detections) >= max_objects:
                    break

    # Convert the result to JSON
    result_json = json.dumps(detections, indent=2)
    return result_json

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

# Example usage:
if __name__ == "__main__":
    #image_path = os.path.expanduser("~/Downloads/test7.png")  # Replace with your image path
    image_path = os.path.expanduser("~/Documents/GitHub/EcomMediaPlayer/MediaPlayerBackend/storage/app/public/uploads/fashion_5.166349_DeekuZeus.png")  # Replace with your image path
    #output_path = os.path.expanduser("~/Downloads/test7_with_detections.png")  # Replace with desired output path
    output_path = '../outputs/img/output_.png'
    
    detections = detect_objects(image_path, max_objects=5)

    # # Convert JSON string to Python list
    # data = json.loads(detections)
    # # Apply the filter
    # filtered_data = filter_overlapping_entries(data, iou_threshold=0.5)
    # # Convert filtered data back to JSON
    # filtered_json = json.dumps(filtered_data, indent=2)
    # print(filtered_json)


    #save_image_with_detections(image_path, output_path, detections)
    save_image_with_detections(image_path, output_path, detections)
    
    print("Detections JSON:")
    print(detections)