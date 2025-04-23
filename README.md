# 🌿 Plant Disease Detection – Smart Farming Using AI

A machine learning model that detects plant diseases through leaf images. Built using the **PlantVillage dataset**, this project demonstrates a practical AI solution for smart farming applications like **drones**, **mobile apps**, and **robotics-based agriculture**.

---

## 📌 Project Overview

**Goal:**  
Detect diseases in plants by analyzing leaf images and classifying them into disease categories for early intervention.

**Use Case:**  
Farmers, agritech companies, or researchers can use this solution to monitor and identify plant diseases through image capture.

---

## 🧠 Problem Statement

Manual inspection of plant diseases is time-consuming and error-prone. With thousands of plant varieties and even more diseases, it's essential to build **automated tools** that assist in identifying infections quickly and accurately.

---

## 🎯 Objectives

- Train a machine learning model using leaf images.
- Classify input images into **healthy** or **diseased** categories.
- Achieve robust accuracy using convolutional neural networks (CNN).
- Ensure the model is lightweight and deployable.

---

## 📂 Dataset

- **Source**: Kaggle – [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes**: 14 Plant types | 300+ Disease categories
- **Format**: RGB images of plant leaves
- **Image Count**: 50,000+ images

---

## ⚙️ Methodology

1. **Image Input Processing**  
   - Image converted to a numerical array
   - Preprocessed using **thresholding** to reduce background noise

2. **Model Architecture**
   - **CNN-based model**
   - Optimizer: **Adam’s Algorithm**
   - Epochs: **32**
   - Accuracy: **86%**

3. **Model Evaluation**
   - Accuracy
   - Confusion Matrix
   - Loss Curve
   - Prediction Confidence

---

## 🚀 Deployment Potential

This model can be deployed in:
- 🌾 Drones for aerial scanning
- 📱 Mobile apps for instant field detection
- 🤖 Farm robots with onboard cameras
- 🌐 Web-based farmer portals

---



