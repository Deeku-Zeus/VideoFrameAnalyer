from flask import Flask, request, jsonify
import os
from model.objectDetection import get_detect_objects_list
from model.classificationFashion import build_transfer_learning_model, preprocess_image, classify_image
from model.classificationColor import get_average_color, classify_color
from model.classificationYoloFashion import detect_objects
from PIL import Image
from utilities.common import generate_unique_hash

app = Flask(__name__)

# Define class names for Fashion MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
num_classes = len(class_names) # 10
# Load the transfer learning model
model = build_transfer_learning_model(num_classes)

# @app.route('/detect', defaults={'object_count':5,'show_objects': False}, methods=['GET'])
@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    object_count = int(data.get('object_count', 5))
    image_path = data.get('image_path', None)
    # Access the uploaded file
    if image_path is None:
        return jsonify({'error': 'path is needed'}), 400

    # file = request.files['image']

    # # Check if the file is present and has a valid filename
    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'}), 400

    # # Load the image file directly using PIL
    # try:
    #     image = Image.open(file)  # type: ignore # Do not use file.read(), just pass the file directly
    # except Exception as e:
    #     return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
    
    image_path = os.path.expanduser("~/Documents/GitHub/EcomMediaPlayer/MediaPlayerBackend/storage/app/public/uploads/{}".format(image_path))
    # Detect objects
    detected_objects = detect_objects(image_path, max_objects=object_count)
    
    # Return detected boxes as JSON response
    return detected_objects

@app.route('/classify', methods=['GET'])
def classify_object():

    # need to send coordinates of yolo detection ( cropped )
    # color classification
    # fashion classification
    return jsonify({"message":"under development !!!!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)  # Make sure to listen on all network interfaces
