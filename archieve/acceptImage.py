from PIL import Image
import numpy as np

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)  # Reshape for the model
    return img_array