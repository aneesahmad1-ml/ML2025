import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

st.title("MNIST Digit Recognition")
st.write("Draw a digit (0–9) and let model predict it.")

MODEL_PATH = "BestModel.h5"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file `BestModel.h5` not found.")
        st.stop()
    return keras.models.load_model(MODEL_PATH)

model = load_model()

canvas = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)


if st.button("Predict"):
    if canvas.image_data is not None:
        img = canvas.image_data[:, :, 0]
        img = Image.fromarray(img).resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28)

        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        st.subheader(f"Predicted Digit: {digit}")
        st.write(f"Confidence: {confidence:.2f}")
