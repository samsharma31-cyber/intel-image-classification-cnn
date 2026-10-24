
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Intel Image Classifier",
    page_icon="🌄",
    layout="centered"
)

st.title("🌄 Intel Image Classification")
st.write("Upload a landscape image to predict its scene category.")

MODEL_PATH = "intel_cnn_model[1].keras"
CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify Image"):
        # Use the same image size and scaling as training.
        image_size = model.input_shape[1:3]
        image = image.resize((image_size[1], image_size[0]))

        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))

        st.success(f"Prediction: {CLASS_NAMES[predicted_index]}")
        st.write(f"Confidence: {predictions[predicted_index] * 100:.2f}%")

        st.subheader("Class probabilities")
        for name, probability in zip(CLASS_NAMES, predictions):
            st.write(f"**{name}:** {probability * 100:.2f}%")
            st.progress(float(probability))
