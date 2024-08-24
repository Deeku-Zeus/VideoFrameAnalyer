# color_classification.py

from sklearn.cluster import KMeans
import numpy as np
from PIL import Image

def get_average_color(image):
    image = image.convert('RGB')
    np_image = np.array(image)
    pixels = np_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=1).fit(pixels)
    return kmeans.cluster_centers_[0]

def classify_color(color):
    # Define a mapping of colors to their names
    color_mapping = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        # Add more colors as needed
    }
    
    min_dist = float('inf')
    closest_color = 'unknown'
    for name, rgb in color_mapping.items():
        dist = np.linalg.norm(np.array(color) - np.array(rgb))
        if dist < min_dist:
            min_dist = dist
            closest_color = name
            
    return closest_color
