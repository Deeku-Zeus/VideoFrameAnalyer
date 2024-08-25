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
    'cyan': [0, 255, 255],
    'magenta': [255, 0, 255],
    'yellow': [255, 255, 0],
    'black': [0, 0, 0],
    'white': [255, 255, 255],
    'gray': [128, 128, 128],
    'silver': [192, 192, 192],
    'maroon': [128, 0, 0],
    'olive': [128, 128, 0],
    'purple': [128, 0, 128],
    'teal': [0, 128, 128],
    'navy': [0, 0, 128],
    'aqua': [0, 255, 255],
    'fuchsia': [255, 0, 255],
    'lime': [0, 255, 0],
    'silver': [192, 192, 192],
    'gold': [255, 215, 0],
    'salmon': [250, 128, 114],
    'coral': [255, 127, 80],
    'turquoise': [64, 224, 208],
    'periwinkle': [204, 204, 255],
    'chocolate': [210, 105, 30],
    'khaki': [240, 230, 140],
    'plum': [221, 160, 221],
    'indigo': [75, 0, 130],
    'orchid': [218, 112, 214],
    'wheat': [245, 222, 179],
    'lavender': [230, 230, 250],
    'lightblue': [173, 216, 230],
    'tan': [210, 180, 140],
    'peachpuff': [255, 218, 185]
    }
    
    min_dist = float('inf')
    closest_color = 'unknown'
    for name, rgb in color_mapping.items():
        dist = np.linalg.norm(np.array(color) - np.array(rgb))
        if dist < min_dist:
            min_dist = dist
            closest_color = name
            
    return closest_color
