Plant Leaf Disease Detection using CNN
-Project Overview

This project focuses on automatic classification of plant leaf diseases using image-based deep learning techniques. The system analyzes images of plant leaves and identifies whether they are healthy or affected by a specific disease.

The goal is to support farmers and agricultural experts by providing a fast, cost-effective, and accurate tool for early disease detection, which helps in improving crop yield and reducing losses.

🔍 Problem Statement

Traditional disease detection in crops depends on manual inspection.

It is time-consuming, costly, and prone to error.

Lack of timely diagnosis leads to reduced productivity and financial losses.

An automated solution is required to make disease classification scalable, reliable, and efficient.

-Objectives

Build a deep learning-based image classification system for plant diseases.

Train and evaluate a CNN model on multi-class crop datasets.

Provide an easy-to-use interface for users to upload leaf images and receive predictions.

Achieve high accuracy with reduced false detections.

⚙️ Technologies Used

Programming Language: Python

Frameworks/Libraries: PyTorch, TensorFlow/Keras, Albumentations, OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn

Interface: Gradio

Tools: Jupyter Notebook, GitHub

Combined to cover 40+ plant diseases across multiple species.

Publicly available on Kaggle PlantVillage

🛠 System Design
Data Preprocessing: Image resizing, normalization, augmentation (rotation, brightness, flipping).

Model Architecture:

Dual-branch model combining a custom CNN and pretrained ResNet34.

Feature fusion for improved classification performance.

Training:
Optimizer: AdamW

Loss Function: CrossEntropyLoss

Techniques: Label smoothing, early stopping, mixed precision training

Evaluation: Accuracy, confusion matrix, and classification reports.

📊 Results
Training Accuracy: ~94%

Validation Accuracy: ~89%

Test Accuracy: ~87%

Inference Speed: 0.1 sec per image

-Key Features
Multi-class plant disease detection.

Robust model using CNN + ResNet34.

High accuracy with optimized training pipeline.

Simple Gradio interface for real-time predictions.

-Future Scope
Deploy on mobile and edge devices for offline usage.

Integrate with drone imagery and IoT sensors.

Expand datasets for more crops and disease types.

Model optimization with pruning/quantization for faster inference.
