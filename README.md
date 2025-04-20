
# 💤 Drowsiness Detection System

This **Drowsiness Detection System** is a deep learning-based web application built with Streamlit that predicts drowsiness by analyzing eye states through images and video input. It uses a CNN model trained on custom image data and provides a simple interface to upload media and view real-time predictions.

---

## 🚀 Features

- 🖼️ Predict drowsiness from eye images
- 🎥 Analyze drowsiness in real-time from video
- 📊 Interactive Streamlit web interface
- 🧠 Deep learning-based backend using a CNN model
- 📁 Includes training/testing dataset and media samples

---

## 🛠️ Tech Stack

| Technology     | Purpose                    |
|----------------|----------------------------|
| Python         | Core programming language  |
| TensorFlow     | Deep learning model        |
| OpenCV         | Video and image processing |
| Streamlit      | Web interface              |
| NumPy, PIL     | Data handling & utilities  |

---

## 📂 Files Included

- `Backend.py` – Script to handle image-based drowsiness predictions
- `DDSfront.py` – Streamlit frontend for image/video input
- `train_model.py` – Training script for the CNN model
- `model.h5` – Trained CNN model
- `dataset/` – Contains training and testing images
- `samples/` – Sample photos and video for demo

---

## ⚙️ How to Run

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the web app using Streamlit:
   ```bash
   streamlit run DDSfront.py
   ```

3. Upload an image or video from the interface to get predictions.

---

## 📁 Dataset and Media

The dataset includes labeled eye-state images categorized as `open` or `closed`. It also includes:

- 1 sample video
- 2–3 sample images
- Separate train and test directories

> You can modify or expand the dataset as needed to improve model performance.

---

## 📌 Future Improvements

- Real-time webcam drowsiness detection
- Alerts for drowsy state detection
- Expand dataset with varied lighting conditions

---

## 👩‍💻 Author

**Sri Harshita Prata**

Built with a focus on leveraging deep learning for safety-focused applications in real-world scenarios.
