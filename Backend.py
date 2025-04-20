import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("model.h5")

# Function to preprocess the image (same as in training)
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image. Please check the file path.")
        exit()
    img = cv2.resize(img, (24, 24))
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Function to predict if the person is drowsy or alert
def predict_drowsiness(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    
    # Debugging raw probabilities
    print(f"Raw Prediction Output: {prediction}")
    print(f"Probability of Drowsy: {prediction[0][0]:.4f}")
    print(f"Probability of Alert: {prediction[0][1]:.4f}")
    
    # Predict status based on the highest probability
    status = "Drowsy" if prediction[0][1] < 0.6 else "Alert"
    return status, prediction


# Upload and predict the image
image_path = "C:/Projects/DDS/Scripts/drowsy1.jpg"

# Predict the drowsiness status
status, prediction = predict_drowsiness(image_path)

# Display the image and result
img = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()

print(f"Prediction: {status}")
