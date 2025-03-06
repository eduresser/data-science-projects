# Anomaly Detection in ECG Data

## Overview

This project implements an unsupervised learning approach to identify anomalies in electrocardiogram (ECG) data using autoencoders. The dataset is available [here](https://www.kaggle.com/datasets/devavratatripathy/ecg-dataset/data). The system learns the pattern of normal ECG signals and identifies anomalies through reconstruction errors that exceed an optimized threshold.

## Technical Approach
The solution consists of two main components:

1. **Autoencoder:** A neural network that learns to reconstruct normal ECG data
2. **Weighted Reconstruction Error:** A mechanism that assigns importance weights to different parts of the ECG signal

## Dependencies

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy

## How It Works

### Data Preprocessing

- The ECG data is scaled to a range of [-1, 1] to facilitate training
- Training is performed primarily on normal ECG samples

### Model Architecture

- The autoencoder uses a gradual compression strategy with dimensions [140, 96, 48, 24]
- Dropout (0.1) and BatchNormalization are used to prevent overfitting
- The model uses tanh as the final activation function

### Anomaly Detection Process

- Train the autoencoder on normal ECG data
- Calculate reconstruction errors for all samples
- Apply feature importance weighting to emphasize critical regions of the ECG
- Determine optimal threshold using ROC curve and Youden index
- Classify signals with reconstruction errors above the threshold as anomalies

### Results

The model achieves excellent performance metrics on the test set:

- Accuracy: 97.2%
- Precision: 95.6%
- Recall: 99.8%
- F1-Score: 97.7%

The high recall value is particularly important in medical contexts, as it indicates that the model rarely fails to identify an anomalous ECG (few false negatives).

## Conclusion
This approach demonstrates the effectiveness of autoencoders for unsupervised anomaly detection in ECG data. The weighted error mechanism improves discrimination between normal and anomalous patterns.