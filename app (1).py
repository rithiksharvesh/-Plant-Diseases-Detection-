# app.py

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model('plant_disease_model.h5')
class_names = ['Apple Scab', 'Corn Blight', 'Grape Black Rot', 'Healthy', 'Potato Late Blight']  # Example classes

# App interface
st.title("🌿 Plant Disease Detection")
uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read and preprocess image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image_resized = cv2.resize(image, (128, 128))
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.success(f"Predicted Disease: {predicted_class}")
