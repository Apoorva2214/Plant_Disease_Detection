import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import base64

# Define working directory and paths
model_path = r"C:/Users/DELL/Desktop/plant-disease-prediction/app/plant_disease_prediction_model .h5"
class_indices_path = r"C:/Users/DELL/Desktop/plant-disease-prediction/app/class_indices.json"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class names
with open(class_indices_path) as f:
    class_indices = json.load(f)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path).convert('RGB')  # Convert image to RGB
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Function to add local background image
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .main {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        .stButton>button {{
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px 24px;
            cursor: pointer;
        }}
        .stButton>button:hover {{
            background-color: #45a049;
        }}
        .stTitle {{
            font-family: "Arial Black", Gadget, sans-serif;
            color: black;  /* Black color for title */
        }}
        .stFileUploader label {{
            color: #FFD700;  /* Gold color */
            font-size: 16px;
            font-family: Arial, sans-serif;
        }}
        .stSuccess {{
            color: black;  /* Black color for output */
            font-size: 24px;  /* Increase the size of the output text */
            font-weight: bold;  /* Make the text bold */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set local background image
set_background_image(r"C:/Users/DELL/Desktop/plant-disease-prediction/app/Screenshot 2024-06-06 185644.png")

# Streamlit App
st.markdown('<h1 class="stTitle">Plant Disease Classifier</h1>', unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((300, 300))  # Increase the size of the uploaded image
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Save the uploaded image to a temporary file
            with open("temp_image.png", "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, "temp_image.png", class_indices)
            st.markdown(f'<p class="stSuccess">Prediction: {str(prediction)}</p>', unsafe_allow_html=True)
            
            # Optionally, remove the temporary file after prediction
            os.remove("temp_image.png")
