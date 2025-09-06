# Durian-Variety-Classification-with-Efficientnet-B0-Architecture---Data-Science

# 🌱 Implementation of Convolutional Neural Network Based on EfficientNet-B0 Architecture for Durian Variety Classification

## 📌 Project Overview
This project implements a Convolutional Neural Network (CNN) based on the **EfficientNet-B0** architecture to classify different varieties of durians.  
The model leverages **transfer learning** from pre-trained ImageNet weights and fine-tunes the network to achieve high classification accuracy.  
The goal is to provide an automated system that can recognize durian varieties from images, which may support research and agricultural applications.

---

## 📂 Dataset
- **Source**: Custom dataset collected from [https://www.kaggle.com/datasets/edenbarua/picture & taking sample photos directly at a durian fruit shop in the Bandar Lampung area].  
- **Total Images**: ~1400 images.  
- **Classes**: 7 varieties of durians (Musang King, D24/Sultan, Golden Phoenix, Sumatra Super, Medan, Bengkulu, and Kota Agung).  
- **Preprocessing**:
  - Image resizing to 224×224 pixels.  
  - Normalization (scaling pixel values to [0,1]).  
  - Data augmentation (rotation, zoom, flip, shift).  

---

## ⚙️ Methodology
- **Architecture**: EfficientNet-B0 (baseline network).  
- **Approach**: Transfer learning (ImageNet).  
- **Framework**: TensorFlow / Keras.  
- **Hyperparameters**:
  - Optimizer: Adam.  
  - Learning Rate: 0,001.  
  - Batch Size: 16.  
  - Epochs: 20.  

---

## 📊 Results
- **Best Accuracy Training**: 99,89% (validation).
- - **Best Accuracy Data Testing**: 98%.  
- **Metrics**: Accuracy, Precision, Recall, F1-score.  
- **Evaluation**:  
  - Confusion Matrix.  
  - Classification Report.  




