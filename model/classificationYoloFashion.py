import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("valentinafeve/yolos-fashionpedia")
model = AutoModelForObjectDetection.from_pretrained("valentinafeve/yolos-fashionpedia")

# Access the id2label mapping from the model configuration
id2label = model.config.id2label

# Print the class names
for id, label in id2label.items():
    print(f"Class ID: {id} -> Class Name: {label}")
    
# Define the selected class names
selected_classes = [
    "shirt, blouse",  # Class ID: 0
    "top, t-shirt, sweatshirt",  # Class ID: 1
    "sweater",  # Class ID: 2
    #"cardigan",  # Class ID: 3
    "jacket",  # Class ID: 4
    "vest",  # Class ID: 5
    "pants",  # Class ID: 6
    "shorts",  # Class ID: 7
    "skirt",  # Class ID: 8
    "coat",  # Class ID: 9
    "dress",  # Class ID: 10
    #"jumpsuit",  # Class ID: 11
    #"cape",  # Class ID: 12
    "glasses",  # Class ID: 13
    "hat",  # Class ID: 14
    "headband, head covering, hair accessory",  # Class ID: 15
    "tie",  # Class ID: 16
    "glove",  # Class ID: 17
    "watch",  # Class ID: 18
    "belt",  # Class ID: 19
    "leg warmer",  # Class ID: 20
    "tights, stockings",  # Class ID: 21
    "sock",  # Class ID: 22
    "shoe",  # Class ID: 23
    "bag, wallet",  # Class ID: 24
    "scarf",  # Class ID: 25
    "umbrella",  # Class ID: 26
    "hood",  # Class ID: 27
    # "collar",  # Class ID: 28
    # "lapel",  # Class ID: 29
    # "epaulette",  # Class ID: 30
    # "sleeve",  # Class ID: 31
    # "pocket",  # Class ID: 32
    # "neckline",  # Class ID: 33
    # "buckle",  # Class ID: 34
    # "zipper",  # Class ID: 35
    # "applique",  # Class ID: 36
    # "bead",  # Class ID: 37
    # "bow",  # Class ID: 38
    # "flower",  # Class ID: 39
    # "fringe",  # Class ID: 40
    # "ribbon",  # Class ID: 41
    # "rivet",  # Class ID: 42
    # "ruffle",  # Class ID: 43
    # "sequin",  # Class ID: 44
    # "tassel"  # Class ID: 45
] # Replace with your desired class names

# Load the image to be processed
downloads_folder = os.path.expanduser("~/Downloads")
image_filename = "test2.png"  # Replace with your actual file name
image_path = os.path.join(downloads_folder, image_filename)
image = Image.open(image_path).convert("RGB")

# Process the image for the model
inputs = processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the bounding boxes, labels, and scores
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

# Plot the image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image)
ax = plt.gca()

# Draw bounding boxes and labels on the image
for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
    if score > 0.5:  # Show only predictions with a high confidence score
        label_text = model.config.id2label[label.item()]
        if label_text in selected_classes:
            box = box.tolist()
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            plt.text(xmin, ymin, f"{label_text}: {score:.2f}", color="white", fontsize=12, backgroundcolor="red")

plt.axis("off")
plt.show()