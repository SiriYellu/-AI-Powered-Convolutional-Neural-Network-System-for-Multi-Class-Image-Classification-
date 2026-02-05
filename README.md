# 🌾 AI-Powered Rice Leaf Disease Classification

An end-to-end deep learning system for multi-class classification of rice plant leaf diseases using CNNs and Vision Transformers. This project compares multiple architectures under identical conditions to identify the most effective model for agricultural disease detection.

---

## 📌 Overview

Rice is a critical global food crop, and early detection of leaf diseases is essential to prevent yield loss. Manual identification is often inconsistent and time-consuming. This project builds an automated image classification system using deep learning to detect three common rice leaf diseases:

- Bacterial Blight  
- Brown Spot  
- Leaf Smut  

Multiple architectures were trained and evaluated to determine the most reliable and efficient approach for real-world agricultural deployment.

---

## 🎯 Objectives

- Build an automated image-based disease classification system  
- Compare CNN, transfer learning, and transformer models  
- Evaluate performance using standardized metrics  
- Identify models suitable for real-time and field deployment  

---

## 🧠 Models Implemented

Six deep learning architectures were trained under identical experimental conditions:

1. Custom VGG-like CNN  
2. VGG19 (Transfer Learning)  
3. MobileNetV3 Small  
4. ResNet50  
5. InceptionV3  
6. Tiny Vision Transformer (ViT)  

Each model was adapted for 3-class classification and evaluated using the same dataset and training pipeline.

---

## 📂 Dataset

- Total images: **4,684 labeled rice leaf images**  
- Classes: 3 disease categories  
- Train/Validation split: **80/20**  
- Training images: **3,748**  
- Validation images: **936**  
- Image size: **224 × 224 pixels**  

Images were normalized and augmented using flipping, rotation, zooming, and contrast adjustments to improve generalization.

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Pixel normalization (0–1 scaling)  
- Resizing to 224×224  
- Augmentation for robustness  

### 2. Training Setup
- Optimizer: Adam  
- Loss: Sparse categorical cross-entropy  
- Batch size: 32  
- GPU-enabled training environment  

### 3. Training Epochs
- CNN models: 10 epochs  
- VGG19: 20 epochs  
- Vision Transformer: 15 epochs  

### 4. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

All models were evaluated on the same validation dataset to ensure fair comparison.

---

## 📊 Results

### Validation Accuracy

| Model | Accuracy (%) |
|------|--------------|
| Tiny Vision Transformer | **97.86** |
| MobileNetV3 Small | 97.65 |
| Custom CNN | 96.69 |
| InceptionV3 | 95.51 |
| VGG19 | 80.66 |
| ResNet50 | 64.42 |

The Vision Transformer achieved the best performance, followed closely by MobileNetV3 and the custom CNN.

### Key Observations

- Transformers capture global spatial patterns effectively  
- Lightweight CNNs perform well with lower computation  
- Frozen transfer learning models struggled to adapt  
- Multi-scale architectures improved texture recognition  

---

## ✅ Advantages

- High classification accuracy (>97% for top models)  
- Efficient lightweight models for mobile deployment  
- Fair architectural comparison under controlled conditions  
- Potential for real-time agricultural diagnostics  

---

## ⚠️ Limitations

- Dataset captured under controlled environments  
- Limited real-world variability  
- No interpretability methods (e.g., Grad-CAM)  
- Transformer models require higher computation  

---

## 🚀 Future Work

- Fine-tune pretrained backbones  
- Use real-field datasets  
- Add explainability techniques  
- Optimize models for mobile/edge deployment  
- Deploy as a farmer-friendly mobile app  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow  
- Deep Learning (CNN + Vision Transformers)  
- GPU Training Environment  

---

## 👥 Team

- **Siri Yellu** – Team Lead, preprocessing, augmentation, InceptionV3 training  
- **Pranay Kumar Peddi** – ResNet50 & VGG19  
- **Akshay Krishna Varma Buddharaju** – Custom CNN, ViT, MobileNetV3  

Department of Computer Science, Kennesaw State University

---

## 📌 Applications

- Smart agriculture  
- Mobile disease detection tools  
- Precision farming systems  
- AI-assisted crop monitoring  

---

## 📜 License

For academic and research purposes only.

---

## 📎 Citation

**“An AI-Powered Convolutional Neural Network System for Multi-Class Image Classification of Rice Plant Leaf Diseases.”**
