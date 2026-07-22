# Alzheimer Disease Classification

An image-based deep-learning app that classifies brain MRI scans into Alzheimer's dementia stages. A convolutional neural network trained in Keras is served through a Streamlit web interface where users can upload an image and get a prediction.

## Overview

The app loads a pre-trained Keras model, preprocesses an uploaded MRI image (grayscale, resized to 128×128, normalized), and predicts the dementia stage along with a confidence score.

## Classes

- Non Dementia
- Very Mild Dementia
- Mild Dementia
- Moderate Dementia

## Tech Stack

- Python
- TensorFlow / Keras (CNN model)
- Streamlit (web interface)
- OpenCV & Pillow (image preprocessing)
- NumPy

## Project Structure

```
.
├── app.py                              # Streamlit app for uploading and classifying images
├── basic_alzheimers'_classification.py  # Model training script (originally a Colab notebook)
└── model_kaggle_alzheimer.h5           # Trained Keras model (required to run the app)
```

## Getting Started

```bash
pip install streamlit tensorflow opencv-python pillow numpy
streamlit run app.py
```

Upload a brain MRI image (JPG/PNG) to receive a predicted dementia stage and confidence.

> Note: The trained model file (`model_kaggle_alzheimer.h5`) must be present in the project directory for the app to run.

## Disclaimer

This project is for educational and research purposes only. It is not a medical device and must not be used for diagnosis or clinical decisions.
