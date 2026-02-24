import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

model = tf.keras.models.load_model("tumor_model.keras")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI image to classify tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    img = image.resize((300, 300))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader(f"Prediction: {predicted_class}")
    st.progress(int(confidence))
    st.write(f"Confidence: {confidence:.2f}%")