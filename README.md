# Image Captioning and Day Time prediction
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)

# About
A project to generate automated captions of the image and predict the time of the day. The image processing is done through visual space based combination of **CNN** and **LSTM**.
The image captioning is done through **RNN**. For predicting the time of the day we have used a k-means clustering model which calculate luminance values to calculate tonal distribution of an image.

# Dataset
Due to limitation of processing power we have used **Flick8k** dataset for training and testing. It is small but diverse dataset consisting  of 8000 images.

# Accuracy Acheived

**BLEU Score** : 62.82
