# fashion_transfer_learning.py

from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

def load_model():
    model_name = "tzhao3/vit-L-DeepFashion"
    processor = AutoImageProcessor.from_pretrained("tzhao3/vit-L-DeepFashion")
    model = AutoModelForImageClassification.from_pretrained("tzhao3/vit-L-DeepFashion")
    model.eval()
    return model, processor

def load_model_from_local(model_dir):
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    except ValueError:
        feature_extractor = AutoImageProcessor.from_pretrained(model_dir)
    
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()
    return model, feature_extractor

def preprocess_image(image, feature_extractor):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs

def classify_image(model, img_tensor):
    with torch.no_grad():
        outputs = model(**img_tensor)
        logits = outputs.logits
        predicted_class_index = logits.argmax(-1).item()
    return predicted_class_index
