# Enhanced Image Classification and Segmentation for Security Systems

## Project Title
**Enhanced Image Classification and Segmentation for Security Systems**

## Description
This project aims to develop a robust image classification and segmentation system designed to identify and locate potential threats in images. By leveraging a combination of machine learning and deep learning techniques, the system classifies images as safe or unsafe and further segments unsafe images to pinpoint specific objects like guns, knives, and shurikens.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Testing and Evaluation](#testing-and-evaluation)
- [Performance Metrics](#performance-metrics)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Introduction
With increasing security concerns, there is a critical need for advanced image processing systems capable of detecting and identifying dangerous objects. This project addresses this need by combining Support Vector Machines (SVM) for classification and U-Net for segmentation, providing a comprehensive solution for security image analysis.

## Features
- **Image Classification**: Classifies images as safe or unsafe.
- **Image Segmentation**: Segments unsafe images to identify specific objects such as guns, knives, and shurikens.
- **Performance Metrics**: Calculates overall accuracy, confusion matrix, and Dice coefficient for model evaluation.

## Data Preprocessing
Images are resized to a uniform size for consistency and efficiency in feature extraction and model training.

## Feature Extraction
- **HOG Features**: Captures edge and texture information.
- **Fourier Transform Features**: Analyzes the frequency components of the images.
- **CNN Features**: Utilizes a pre-trained VGG16 model for deep learning feature extraction.

## Model Training
- **SVM**: Trained on combined features (HOG, Fourier, CNN) to classify images.
- **U-Net**: Trained on unsafe images to segment and identify specific objects.

## Testing and Evaluation
Test images are classified and segmented using the trained models. Predicted masks are saved for further analysis, and the performance of the models is evaluated using various metrics.

## Performance Metrics
- **Overall Accuracy**: Measures the percentage of correctly classified images.
- **Confusion Matrix**: Provides insights into classification performance for each category.
- **Dice Coefficient**: Evaluates the overlap between predicted and true segmentations.

## Results
The system demonstrates high accuracy in classifying and segmenting images, effectively identifying dangerous objects within images.

## Contributors
Muhammad Abdullah Ubaidullah - (https://github.com/Dev-Abs)

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Dev-Abs/Enhanced-Image-Classification-and-Segmentation-for-Security-Systems/blob/main/LICENSE.md) file for details.
