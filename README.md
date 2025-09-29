# Durian-Variety-Classification-with-Efficientnet-B0-Architecture---Data-Science

# 🍈 Implementation of Convolutional Neural Network Based on EfficientNet-B0 Architecture for Durian Variety Classification

## 📌 Project Overview
This project focuses on building a **Convolutional Neural Network (CNN)** model based on the **EfficientNet-B0 architecture** to classify images of **durian varieties**.  
Durian is a tropical fruit with many local varieties, and accurate classification is important for agricultural management, research, and commerce.  

The study investigates how model design, training configuration, and hyperparameter tuning affect classification accuracy.

---

## ❓ Problem Statement
1. How to develop a Deep Learning model (CNN with EfficientNet-B0 architecture) for durian variety classification?  
2. How is the performance of CNN EfficientNet-B0 in classifying durian varieties?  
3. How do architectural configurations and hyperparameter settings (learning rate, batch size, dense layers, epochs) affect model performance?  

---

## 🎯 Research Objectives
1. Develop a durian classification model using **CNN EfficientNet-B0** for image-based classification.  
2. Analyze and evaluate the performance of the EfficientNet-B0 model in distinguishing multiple durian varieties.  
3. Conduct **hyperparameter tuning** (learning rate, batch size, dense layer units, and number of epochs) to find the optimal configuration for highest accuracy.  

---

## 📋 Scope & Limitations
- The study is limited to **EfficientNet-B0 CNN architecture** (no comparison with other architectures).  
- The focus is on **model training and evaluation**, not on deployment.  
- Dataset covers **7 durian varieties**, not all existing varieties.  

---

## 🛠️ Methodology
1. **Dataset**: Images of 7 durian varieties (preprocessed and augmented).  
2. **Model**: CNN with EfficientNet-B0 backbone.  
3. **Hyperparameters tested**:
   - Learning rate: `0.001`, `0.0001`, `0.00001`  
   - Batch size: `16`, `32`, `64`  
   - Dense layer units: `64`, `128`, `256`, `(64,128)`, `(64,128,256)`  
   - Epochs: `10`, `20`, `30`  
4. **Evaluation Metrics**:  
   - Accuracy  
   - Precision, Recall, F1-Score (Macro & Weighted average)  
   - Confusion Matrix  

---

## 🔬 Experiments & Results
- Conducted multiple experiments with different hyperparameter settings.  
- The best performing configuration achieved **highest accuracy** with tuned hyperparameters.  
- Results highlight that **learning rate and dense layer units** had the most significant impact on performance.  

*(You can add result tables or graphs here, e.g., accuracy vs epochs, confusion matrix heatmap, etc.)*  

---

## 📊 Example Results
| Configuration | Accuracy | F1-Score |
|---------------|----------|----------|
| LR=0.001, BS=32, Dense=(128), Epoch=20 | 95.3% | 0.95 |
| LR=0.0001, BS=64, Dense=(64,128), Epoch=30 | **97.0%** | **0.97** |
| LR=0.00001, BS=16, Dense=(64,128,256), Epoch=30 | 96.2% | 0.96 |


---

## 📦 Tech Stack
- **Programming**: Python  
- **Deep Learning Framework**: TensorFlow / Keras  
- **Data Analysis**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **Model**: EfficientNet-B0 pretrained weights (Transfer Learning)  

---

## 📂 Repository Structure
📦 durian-cnn-efficientnetb0
┣ 📂 data/ # Dataset (images of durian varieties)
┣ 📂 notebooks/ # Jupyter notebooks for EDA & experiments
┣ 📂 models/ # Saved models & checkpoints
┣ 📜 train.py # Training script
┣ 📜 evaluate.py # Evaluation script
┣ 📜 requirements.txt # Dependencies
┗ 📜 README.md # Project documentation



