# data_preprocessing.py

import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array

def preprocess_image(image_path, target_size=(128, 128)):
    # Read and resize image
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to remove background noise
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Convert back to RGB and normalize
    image_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    image_array = img_to_array(image_rgb) / 255.0
    
    return image_array
