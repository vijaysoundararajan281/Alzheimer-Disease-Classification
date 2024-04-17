import streamlit as st
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

# Load the model
model = load_model('model_kaggle_alzheimer.h5')

# Function to preprocess the image
def preprocess_image(img):
    if len(img.shape) > 2:  # Check if image has more than one channel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale if it has multiple channels
    img = cv2.resize(img, (128, 128))  # Resize image to match model input size
    img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    return img

# Function to predict class label
def predict_image_class(img, model):
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    prediction = model.predict(img)
    return prediction

# Streamlit app
st.title('Alzheimer Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image file
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = np.array(img)
    img = preprocess_image(img)

    # Make prediction
    prediction = predict_image_class(img, model)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx] * 100

    # Display prediction
    classes = ['Mild Dementia', 'Moderate Dementia', 'Non Dementia', 'Very Mild Dementia']  # Define your class labels
    st.write(f'Predicted Class: {classes[class_idx]}')
    st.write(f'Confidence: {confidence:.2f}%')
