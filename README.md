# Digits Recognition (Machine Learning)
## Project Overview

This project focuses on handwritten digit recognition using classical Machine Learning algorithms.
The goal is to classify digits (0–9) from small grayscale images by applying a clean and interpretable ML pipeline.

Unlike deep learning–heavy solutions, this project emphasizes:

Data understanding

Feature scaling

Model comparison

Metric-based evaluation

## Problem Description

Handwritten digit recognition is a multi-class classification problem where each input sample belongs to one of 10 classes (0–9).

Each digit is represented as an 8×8 grayscale image, flattened into numerical features.

## Dataset

Dataset: sklearn.datasets.load_digits

Samples: 1,797

Classes: 10 (digits 0–9)

Features: 64 pixel intensity values

Image Shape: (8 × 8)

The project also includes data exploration and visualization to understand the dataset structure.

## Dataset Exploration

Before training, the dataset is inspected to understand:

Target labels

Feature dimensions

Image representation

A sample image is visualized to confirm the relationship between:

Pixel values

Image structure

True label

This step ensures transparency and interpretability of the data.

# Data Preprocessing
## Train / Test Split

70% training

30% testing

train_test_split(test_size=0.3)

## Feature Scaling

MinMaxScaler is applied to normalize features into the range [0, 1]

This step is critical for distance-based and neural models

MinMaxScaler()

# Models Implemented

The following Machine Learning models are trained and evaluated:

Model	Description
K-Nearest Neighbors (KNN)	Distance-based classifier
Support Vector Machine (SVM)	Linear kernel classifier
Random Forest (RF)	Ensemble tree-based model
Artificial Neural Network (ANN)	Multi-layer perceptron

All models are trained on the same processed data to ensure a fair comparison.

## Evaluation Metrics

Each model is evaluated using standard multi-class classification metrics:

Accuracy – Overall correctness

Precision (Weighted) – Prediction reliability across all classes

Recall (Weighted) – Ability to correctly identify all classes

A shared evaluation function ensures consistent metric computation across models.

## Performance Visualization

Two bar charts are generated:

- Training Accuracy

Used to analyze model fitting behavior.

- Testing Accuracy

Used to evaluate generalization performance on unseen data.

These plots help detect:

Overfitting

Underfitting

Relative model strength

## Experimental Design

Same dataset

Same preprocessing pipeline

Same evaluation metrics

Only the model architecture changes

This design ensures a controlled and scientifically valid comparison.


## Skills Demonstrated

Dataset exploration and visualization

Feature scaling and preprocessing

Multi-class classification

Metric-based evaluation

Model comparison and analysis

Clean and interpretable ML workflow

## Project Structure
Digits-Recognition/
│
├── main.py
├── README.md
└── requirements.txt

## Technologies Used

Python

Scikit-learn

NumPy

Matplotlib



Compare with CNN-based approaches

## Author

Nisar Ahmad Zamani
Machine Learning | Artificial Intelligence
