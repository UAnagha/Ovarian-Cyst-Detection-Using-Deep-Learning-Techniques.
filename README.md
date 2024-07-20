# Ovarian Cyst Detection using Deep Learning

## Abstract
This project involves developing a Convolutional Neural Network (CNN) based model named OvarianNet for classifying ultrasound images of ovaries as normal or containing cysts. The goal is to aid in the early and accurate detection of ovarian cysts, which are fluid-filled sacs that develop on or within the ovaries. Manual identification of cysts in ultrasound images can be challenging, so this automated approach aims to improve detection accuracy and efficiency.

The research utilizes a dataset of labeled ultrasound images that have been pre-processed to normalize them and augmented to increase diversity. The CNN architecture includes convolution layers for feature extraction and fully connected layers for classification. Various models, including SequentialNet, DenseNet169, ResNet50, and MobileNet, were compared to evaluate their performance.

## Introduction
Ovarian cysts, which are fluid-filled sacs located within the ovaries, are frequently encountered in gynecology. While most are benign, some pose a risk of malignancy. Detecting ovarian cysts early and accurately is crucial for optimal patient outcomes. Ultrasound imaging is widely used for ovarian visualization, but manually identifying cysts can be challenging due to their varied appearance and subjective interpretation.

The CNN model developed in this project demonstrated high accuracy in distinguishing between normal ovaries and those containing cysts on unseen test data. Specific performance metrics, including F1 score, accuracy, sensitivity, specificity, and recall, are presented.

This research highlights the potential of CNNs for automated ovarian cyst detection in ultrasound images, providing a valuable tool to assist healthcare professionals in early diagnosis and ultimately improving patient outcomes.

**Keywords**: Ovarian cyst detection, deep learning, Convolutional Neural Network, OvarianNet, SequentialNet, MobileNet, ResNet50, DenseNet169.

## Dataset
The dataset used for this project consists of labeled ultrasound images of ovaries. The dataset was compiled from various sources and pre-processed to ensure consistency and quality. Key aspects of the dataset include:

- **Source**: The dataset was created on Kaggle by combining data from multiple sources on the internet.
- **Pre-processing**: Images were normalized to ensure uniformity and pre-processed to enhance model performance.
- **Augmentation**: Data augmentation techniques were applied to increase the diversity of the dataset and improve model robustness.

The dataset includes both normal and cystic ovarian images, which were used to train and evaluate the CNN model.

## Features
- Custom CNN architecture (OvarianNet) achieving 86% training accuracy and 85% testing accuracy.
- Comparative analysis with models like SequentialNet, DenseNet169, ResNet50, and MobileNet.
- Visualization of training and testing accuracies using Matplotlib and Seaborn.
- Detailed performance metrics including F1 score, accuracy, sensitivity, specificity, and recall.

## Technologies Used
- Python
- PyTorch
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

- Install PyTorch
-pip install torch torchvision torchaudio

- Install Matplotlib, Seaborn, Scikit-learn.
-pip install matplotlib seaborn scikit-learn


## Model Architecture
The custom CNN architecture is designed to optimize detection accuracy while maintaining computational efficiency. The architecture includes several convolutional layers, max-pooling layers, and fully connected layers, with ReLU activation functions and dropout for regularization.

## Comparative Analysis
This project includes a comparison of the custom CNN model with the following architectures:
- SequentialNet
- DenseNet169
- ResNet50
- MobileNet

The comparative analysis covers accuracy metrics, computational performance, and model robustness. The custom CNN model demonstrated high accuracy in distinguishing between normal ovaries and those containing cysts.

## Data Preprocessing and Augmentation
Various data preprocessing techniques were applied, including normalization and augmentation, to enhance model robustness and generalizability.

## Visualization
Matplotlib and Seaborn were used for visualizing training and testing accuracies, as well as for plotting the comparative analysis graph.

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request.
