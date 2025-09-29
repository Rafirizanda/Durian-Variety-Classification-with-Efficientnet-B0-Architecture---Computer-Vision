# Durian-Variety-Classification-with-Efficientnet-B0-Architecture---Data-Science

# 🍈 Implementation of Convolutional Neural Network Based on EfficientNet-B0 Architecture for Durian Variety Classification

## 📖 Project Overview

Durian (*Durio zibethinus Murr*) is one of the most popular tropical fruits in Southeast Asia with high economic value. In Indonesia, durian production has shown a **consistent increase over the past five years**, reaching nearly **2 million tons in 2024** according to official statistics (BPS). Durian is not only a key agricultural commodity but also contributes to **agrotourism and international trade**, making its quality and varietal classification crucial for farmers, sellers, and consumers.

However, challenges remain in the **accurate identification of durian varieties**. Currently, classification is often performed manually, requiring expert knowledge and leaving room for **market manipulation** (e.g., selling common durian as premium “Musang King”). Such misclassification leads to consumer distrust and price instability in the market.  

To address these issues, this project leverages **Deep Learning and Image Processing** to build an automated durian variety classification system. Specifically, we implement a **Convolutional Neural Network (CNN) using EfficientNet-B0 architecture**, which is well-known for its **balance between accuracy and computational efficiency**.  

EfficientNet-B0 offers several advantages:
- Lightweight architecture (5.3M parameters, 0.39B FLOPs) → suitable for real-time applications.  
- Strong performance in fruit image datasets (up to 99%+ accuracy).  
- Supports **transfer learning with ImageNet pre-trained weights**, reducing training time and improving accuracy with limited labeled data.  

By experimenting with **hyperparameter tuning (learning rate, batch size, dense layers, epochs)**, this project aims to develop a robust classification model that can accurately recognize **7 durian varieties**. The system is expected to contribute towards:  
- Enhancing transparency in the durian market.  
- Reducing fraud in varietal labeling.  
- Supporting farmers and sellers in maintaining quality assurance.  
- Opening opportunities for **real-time applications** in agriculture and agrotourism.  

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


To evaluate the performance of the **EfficientNet-B0** model for **Durian Variety Classification**, several experiments were conducted by varying key hyperparameters: **number of epochs, learning rate, batch size, and dense layer configuration**.  

---

### 📊 Performance Metrics per Hyperparameter

#### 1. Number of Epochs
| Epoch | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|-----------|
| 10    | 0.97     | 0.97      | 0.97   | 0.97      |
| 20    | 0.97     | 0.98      | 0.97   | 0.98      |
| 30    | 0.96     | 0.97      | 0.96   | 0.96      |
| 40    | 0.96     | 0.97      | 0.96   | 0.96      |

---

#### 2. Learning Rate
| Learning Rate | Accuracy | Precision | Recall | F1-score |
|---------------|----------|-----------|--------|-----------|
| 0.01          | 0.94     | 0.96      | 0.94   | 0.94      |
| 0.001         | 0.97     | 0.97      | 0.97   | 0.97      |
| 0.0001        | 0.96     | 0.97      | 0.96   | 0.96      |

---

#### 3. Batch Size
| Batch Size | Accuracy | Precision | Recall | F1-score |
|------------|----------|-----------|--------|-----------|
| 16         | 0.97     | 0.97      | 0.97   | 0.97      |
| 32         | 0.98     | 0.98      | 0.98   | 0.98      |
| 64         | 0.98     | 0.98      | 0.98   | 0.98      |

---

#### 4. Dense Layer Configuration
| Dense Layer        | Accuracy | Precision | Recall | F1-score |
|--------------------|----------|-----------|--------|-----------|
| 64                 | 0.97     | 0.97      | 0.97   | 0.97      |
| 128                | 0.98     | 0.98      | 0.98   | 0.98      |
| 256                | 0.97     | 0.98      | 0.97   | 0.97      |
| 64, 128            | 0.97     | 0.97      | 0.97   | 0.97      |
| 64, 128, 256       | 0.96     | 0.97      | 0.96   | 0.96      |

---

### ✅ Best Configuration

The best results were obtained with the following configuration:
- **Dense Layer = 128 neurons**  
- **Epoch = 20**  
- **Learning Rate = 0.001**  
- **Batch Size = 16**  


## ✅ Conclusion
1. A classification model for durian varieties was successfully developed using the **EfficientNet-B0 architecture**, achieving up to **99% accuracy** in classifying 7 durian varieties.  
2. The proposed EfficientNet-B0 model demonstrated **high performance** using **transfer learning from ImageNet** with **data augmentation** and **without fine-tuning**, achieving:  
   - Training accuracy: **99.67%**  
   - Validation accuracy: **100%**  
   - Training loss: **0.1383**  
   - Validation loss: **0.1283**  
   The analysis showed that training **without fine-tuning performed better** than fine-tuning, which tended to cause overfitting.  
3. The **best hyperparameter configuration** obtained through tuning consisted of:  
   - Dense layer: **128 neurons**  
   - Epochs: **20**  
   - Learning rate: **0.001**  
   - Batch size: **16**  
   With this setup, the model achieved:  
   - Training accuracy: **99.95%**  
   - Validation accuracy: **99.29%**  
   - Training loss: **0.0699**  
   - Validation loss: **0.0805**
     
---

## 📦 Tech Stack
- **Programming**: Python  
- **Deep Learning Framework**: TensorFlow / Keras  
- **Data Analysis**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **Model**: EfficientNet-B0 pretrained weights (Transfer Learning)  

---



