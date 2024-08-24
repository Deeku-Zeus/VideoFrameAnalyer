# model/fashion_transfer_learning.py

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

def build_transfer_learning_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)
    return img_array

def classify_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class
