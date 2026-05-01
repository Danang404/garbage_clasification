# ♻️ Recyclable Waste Classification with PyTorch & MLflow

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow)

## 📌 Project Overview
This repository contains a robust Deep Learning pipeline to classify recyclable waste into **12 different categories** (Battery, Biological, Brown Glass, Cardboard, Clothes, Green Glass, Metal, Paper, Plastic, Shoes, Trash, White Glass). 

By leveraging **EfficientNet-V2-S**, the model is trained to accurately identify and separate waste, promoting better recycling automation and environmental sustainability.

## ✨ Key Features
- **State-of-the-Art Architecture**: Utilizes `EfficientNet-V2-S` for an optimal balance between accuracy and computational efficiency.
- **Two-Phase Training Strategy**:
  1. **Initial Transfer Learning**: Freezing the base layers to train the custom classifier head.
  2. **Fine-Tuning**: Unfreezing all layers with a lower learning rate (`1e-5`) to achieve maximum accuracy.
- **Experiment Tracking**: Fully integrated with **MLflow** to automatically log parameters, metrics (Train/Val Loss & Accuracy), and model artifacts.
- **Modern Data Augmentation**: Uses `torchvision.transforms.v2` for robust image preprocessing (Random Flip, Rotation, and Resized Cropping).

## 📊 Results & Performance
The model achieves exceptional accuracy across all 12 classes. The confusion matrix on the test set demonstrates highly solid diagonal predictions with minimal misclassifications.
*(Kamu bisa upload gambar confusion_matrix.png kamu ke github, lalu tambahkan gambarnya di sini)*
<!-- ![Confusion Matrix](link_gambar_confusion_matrix_kamu_di_github.png) -->

## 🚀 Tech Stack
- **Framework:** PyTorch & Torchvision
- **Tracking & MLOps:** MLflow
- **Data Handling:** Scikit-Learn, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
