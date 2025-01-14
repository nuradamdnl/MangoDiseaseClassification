import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Cache the model loading function
@st.cache_resource
def load_model():
    # Load the Mango Leaf Disease Classification model
    model = tf.keras.models.load_model('./mango_leaf_disease_model.h5')
    return model

# Function to preprocess the image and make predictions
def predict_class(image, model, class_names):
    # Cast image to float32 and resize to match the input shape of the model
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [180, 180])
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Get predictions
    prediction = model.predict(image)
    confidence = tf.nn.softmax(prediction[0])  # Convert logits to probabilities

    return np.argmax(confidence), confidence

# Load the model
model = load_model()

# Define class names
class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil']

# Streamlit app title
st.title('Mango Leaf Disease Classifier')

# File uploader for user to upload an image
file = st.file_uploader("Upload an image of a mango leaf", type=["jpg", "png"])

# If no file is uploaded, display a waiting message
if file is None:
    st.text('Waiting for an image upload...')
else:
    # Display loading message
    slot = st.empty()
    slot.text('Running inference...')

    # Open the uploaded image
    test_image = Image.open(file)

    # Display the uploaded image
    st.image(test_image, caption="Uploaded Image", width=400)

    # Predict the class of the uploaded image
    pred_class_idx, confidence_scores = predict_class(np.asarray(test_image), model, class_names)

    # Get the predicted class name and its confidence score
    result = class_names[pred_class_idx]
    confidence = confidence_scores[pred_class_idx].numpy() * 100  # Convert to percentage

    # Display the result
    slot.text('Done')
    st.success(f'The image is classified as **{result}** with a confidence of {confidence:.2f}%.')
