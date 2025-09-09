# Plant-leaf-disease-detection
Deep Learning-based CNN model for crop disease detection.
Plant Leaf Disease Classification using CNN
Overview

This project applies Deep Learning (Convolutional Neural Networks) to detect and classify crop leaf diseases from images.
The aim is to provide farmers and agricultural experts with an automated, scalable, and cost-effective solution for early disease detection, improving crop yield and reducing losses.
Problem Statement

Farmers often face difficulty in identifying plant diseases at an early stage.

Manual inspection is time-consuming, error-prone, and not scalable.

Delayed detection results in reduced yield and financial loss.

This project solves the problem by building an AI-powered model that classifies healthy vs diseased leaves and identifies specific disease categories.

Solution

Built a Convolutional Neural Network (CNN) combined with a pretrained ResNet34 for robust feature extraction.

Trained on publicly available datasets such as PlantVillage and Allen & Mufti datasets covering 40+ plant diseases.

Integrated with a Gradio web interface, allowing users to upload a leaf image and instantly get predictions with confidence scores.

⚙️ Tech Stack

Programming Language: Python

Deep Learning Frameworks: PyTorch, TensorFlow/Keras

Libraries: NumPy, Pandas, OpenCV, Albumentations, Matplotlib, Seaborn, Sklearn

Interface: Gradio (for real-time predictions)

Tools: Jupyter Notebook / Google Colab, GitHub

Steps & Workflow

Dataset Preprocessing – Image resizing, normalization, and augmentation.

Model Building – Custom CNN + ResNet34 dual-branch architecture.

Training – Optimized using AdamW, learning rate scheduler, and early stopping.

Evaluation – Accuracy, confusion matrix, and classification report.

Deployment – Real-time testing with Gradio interface.

📊 Results

Training Accuracy: ~94%

Validation Accuracy: ~89%

Test Accuracy: ~87%

Inference speed: 0.1s per image
Dataset
https://www.kaggle.com/datasets/emmarex/plantdisease
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
Author

Ankita Sahoo
B.Tech CSE (AI/ML Specialization) | Sri Sri University
