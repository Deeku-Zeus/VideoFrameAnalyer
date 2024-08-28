# main.py
import os
from model.classificationFashion import build_transfer_learning_model, preprocess_image, classify_image
from model.classificationColor import get_average_color, classify_color
from model.objectDetection import get_detect_objects_list
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Define class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def display_image_with_rectangles(image_path, detected_boxes, class_names, model):
    # Load the image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Display the image
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box in detected_boxes:
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Extract and classify the object
        cropped_img = img.crop((x, y, x + w, y + h))
        
        # Classify the fashion item
        img_array = preprocess_image(cropped_img)
        predicted_class_index = classify_image(model, img_array)
        predicted_class_name = class_names[predicted_class_index]

        # Classify the color
        avg_color = get_average_color(cropped_img)
        color_name = classify_color(avg_color)

        # Display the class and color on the image
        plt.text(x, y - 10, f'Class: {predicted_class_name} {color_name}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        #plt.text(x, y - 25, f'Color: {color_name}', color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))

    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Define number of classes (e.g., 10 for Fashion MNIST)
    num_classes = 10

    # Load the transfer learning model
    model = build_transfer_learning_model(num_classes)

    image_path = ''

    # Provide the path to your test image (for windows )
    #image_path = os.path.join("E:\\SystemFiles", "Pictures", "deeku.jpg")

    # for macOS 
    # Get the path to the Downloads folder
    downloads_folder = os.path.expanduser("~/Downloads")
    # Specify the image filename
    image_filename = "fashion_5.166349_DeekuZeus.png"  # Replace with your actual file name
    # Construct the full path to the image file
    image_path = os.path.join(downloads_folder, image_filename)
    


    # Detect a maximum of 5 objects in the image
    detected_boxes = get_detect_objects_list(image_path, max_objects=5)
    print(detected_boxes)
    
    # Display the image with bounding boxes, classifications, and color information
    #display_image_with_rectangles(image_path, detected_boxes, class_names, model)



