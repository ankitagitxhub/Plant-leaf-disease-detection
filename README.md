Plant Leaf Disease Detection using CNN
-Project Overview

This project is about developing a computer vision model that can detect and classify plant leaf diseases from images.

Agriculture depends heavily on plant health, and farmers often struggle to identify diseases early. Manual inspection is not always reliable and requires expert knowledge, which is not available everywhere.

The purpose of this project is to build an automated system that takes a plant leaf image as input and predicts whether the leaf is healthy or infected, along with the type of disease if infected.

This project demonstrates how image classification can be applied in the agricultural domain to improve productivity, reduce pesticide misuse, and support farmers in decision-making.

-Problem Statement

Farmers often face crop loss due to late detection of diseases.

Manual methods require experts, are slow, and may not be accurate at scale.

A fast, scalable solution is needed for real-time disease detection.

-Objectives

Create a deep learning model to classify leaf images into multiple disease categories.

Train and validate the model on large, diverse datasets.

Provide a user-friendly interface where users can upload an image and instantly get predictions.

Achieve high accuracy and fast inference time so the solution can be practical.

⚙️ Technologies Used

Programming Language: Python

Deep Learning Frameworks: PyTorch, TensorFlow/Keras

Supporting Libraries: NumPy, Pandas, OpenCV, Albumentations, Matplotlib, Seaborn, Scikit-learn

Interface: Gradio (for simple web-based predictions)

Development Tools: Jupyter Notebook, Google Colab, GitHub

-Together covering 40+ disease categories across multiple plant species.
-The dataset was cleaned and preprocessed:

Resized to standard dimensions.

Normalized for pixel scaling.

Augmented using Albumentations (rotation, flip, brightness, zoom) to make the model robust.

Reference: PlantVillage Dataset on Kaggle

🛠 System Design
1. Preprocessing

Data organized into train, validation, and test sets.

Augmentation applied to reduce overfitting and improve generalization.

2. Model Architecture
Dual-Branch Model:

Branch 1: Custom CNN with convolution + pooling layers.

Branch 2: Pretrained ResNet34 for transfer learning.

Outputs from both branches were combined (feature fusion) and fed into fully connected layers for classification.

3. Training Setup
Optimizer: AdamW

Loss Function: CrossEntropyLoss

Batch Size: 16

Epochs: 20+

Techniques: label smoothing, mixed precision training, learning rate scheduler, early stopping

4. Evaluation
Metrics: Accuracy, Confusion Matrix, Precision, Recall, F1-score

Visualization: Training/validation accuracy graphs, loss curves

📊 Results
Training Accuracy: ~94%

Validation Accuracy: ~89%

Test Accuracy: ~87%

Inference Speed: ~0.1 sec per image

The model was able to classify diseases such as early blight, late blight, powdery mildew, leaf spot, rust, and more with strong accuracy.

Training/validation accuracy graphs

Confusion matrix visualization

GUI interface (Gradio screenshot)

Example predictions on sample images

-Features
Multi-class classification (healthy + 40+ disease types).

Dual-branch CNN + ResNet34 architecture for better feature learning.

Augmented dataset for robustness.

User-friendly interface for image upload and instant results.

Public GitHub repo for code and documentation.

-Future Improvements
Optimize the model using pruning and quantization to make it deployable on mobile/IoT devices.

Extend the system for real-time drone imagery in large farms.

Support multiple languages in the interface for rural accessibility.

Expand dataset to cover more crops and region-specific diseases.
