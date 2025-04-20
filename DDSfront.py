import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image  # Import the Image module from PIL

# Set the page config with a custom icon for the browser tab
icon = Image.open("icon.jpg")  
st.set_page_config(page_title="Drowsiness Detection System", page_icon=icon)

# Load the trained model
model = load_model("model.h5")

# Function to preprocess the image (same as in training)
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (24, 24))  # Resize the image to 24x24
    img = np.expand_dims(img, axis=-1)  # Add a channel dimension (grayscale)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img

# Function to predict if the person is drowsy or alert
def predict_drowsiness(image):
    img = preprocess_image(image)
    prediction = model.predict(img)
    
    # Debugging raw probabilities
    print(f"Raw Prediction Output: {prediction}")
    print(f"Probability of Drowsy: {prediction[0][0]:.4f}")
    print(f"Probability of Alert: {prediction[0][1]:.4f}")
    
    # Predict status based on the highest probability
    status = "Drowsy" if prediction[0][1] < 0.6 else "Alert"
    return status, prediction

# Streamlit UI
st.title("Drowsiness Detection System")

# Sidebar Menu for navigation
menu = ["Home", "Image", "Video"]
choice = st.sidebar.selectbox("Select Menu", menu)

if choice == "Home":
    st.subheader("Welcome to the Drowsiness Detection System")
    st.write(""" 
        This system can detect whether a person is drowsy or alert based on an uploaded image or video. 
        \n
        Choose Image or Video from the sidebar to test the system.
    """)

elif choice == "Image":
    st.subheader("Upload Image for Drowsiness Detection")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Predict drowsiness status
        status, prediction = predict_drowsiness(image)
        
        # Display the prediction result
        st.write(f"Prediction: {status}")
        
        # Debug: Show prediction probabilities
        st.write(f"Probability of Alert: {prediction[0][1]:.4f}")
        st.write(f"Probability of Drowsy: {prediction[0][0]:.4f}")

elif choice == "Video":
    st.subheader("Upload Video for Drowsiness Detection")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_video.read())
            temp_file_path = temp_file.name

        # Display the video using Streamlit's video player
        st.write("Playing Video:")
        st.video(temp_file_path)  # Directly display the video file

        # Read the video using OpenCV
        video = cv2.VideoCapture(temp_file_path)

        st.write("Processing video...")

        # Track the overall status from frames
        overall_status = None

        # Process each frame
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Preprocess the frame
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            img_resized = cv2.resize(img_gray, (24, 24))  # Resize to 24x24
            img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
            img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
            img_resized = img_resized / 255.0  # Normalize

            # Predict drowsiness status
            prediction = model.predict(img_resized)
            status = "Alert" if prediction[0][1] >= 0.6 else "Drowsy"

            # Update the overall status (just for the last frame)
            overall_status = status

        # Once the video is processed, display the result
        st.write(f"Final Prediction: {overall_status}")

        # Release the video capture and remove the temporary file
        video.release()
        os.remove(temp_file_path)
