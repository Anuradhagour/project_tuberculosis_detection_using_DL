import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
model = tf.keras.models.load_model(r"c:\Users\Lenovo\Desktop\DL PROJECT\vgg16_tb_model (1).keras")



st.title("🩺 Tuberculosis X-ray Classifier")
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption='Uploaded Image', use_container_width=True)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    st.write("🔴 **TB Detected**" if prediction > 0.5 else "🟢 **Normal**")
