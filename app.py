

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="TerraScan AI",
    page_icon="🌄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- CUSTOM DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0b1220 0%, #101e2d 55%, #142a32 100%);
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 2rem 0 1rem 0;
}
.eyebrow {
    color: #5eead4;
    letter-spacing: 3px;
    font-size: 0.75rem;
    font-weight: 800;
}
.hero h1 {
    font-size: clamp(2.5rem, 5vw, 4.4rem);
    line-height: 1.08;
    font-weight: 800;
    color: #f8fafc;
    margin: 0.7rem 0;
}
.hero p {
    color: #b8c7d8;
    font-size: 1.08rem;
    max-width: 700px;
    line-height: 1.8;
}
.panel {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(148,163,184,0.2);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 0.6rem 0 1rem 0;
}
.metric {
    background: rgba(94,234,212,0.07);
    border: 1px solid rgba(94,234,212,0.2);
    border-radius: 15px;
    padding: 1rem;
    text-align: center;
}
.metric .value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #5eead4;
}
.metric .label {
    color: #b8c7d8;
    font-size: 0.8rem;
}
div.stButton > button {
    background: linear-gradient(90deg, #2dd4bf, #5eead4);
    color: #06201e;
    border: none;
    border-radius: 12px;
    font-weight: 800;
    min-height: 3rem;
    width: 100%;
}
div.stButton > button:hover {
    background: #99f6e4;
    color: #06201e;
    border: none;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.025);
    border: 1px dashed #5eead4;
    border-radius: 14px;
    padding: 1rem;
}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- MODEL SETTINGS ----------
MODEL_PATH = Path(__file__).parent / "intel_cnn_model[1].keras"

CLASS_NAMES = [
    "buildings", "forest", "glacier",
    "mountain", "sea", "street"
]

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">INTELLIGENT SCENE RECOGNITION · CNN</div>
  <h1>TerraScan <span style="color:#5eead4;">AI.</span></h1>
  <p>Understand the world, one image at a time. Upload a landscape or outdoor
  photo and let a custom convolutional neural network identify its scene.</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown('<div class="metric"><div class="value">6</div><div class="label">SCENE CATEGORIES</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric"><div class="value">CNN</div><div class="label">MODEL ARCHITECTURE</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric"><div class="value">AI</div><div class="label">IMAGE ANALYSIS</div></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="panel"><h3>📤 Upload your image</h3><p style="color:#b8c7d8">Choose a JPG, JPEG or PNG image for scene recognition.</p></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your uploaded image", use_container_width=True)
        except Exception:
            st.error("This image could not be opened. Please try another file.")
            st.stop()

    classify = st.button("✨  Analyze Scene", use_container_width=True)

with right:
    st.markdown('<div class="panel"><h3>🧠 Prediction results</h3><p style="color:#b8c7d8">Your predicted scene and class probabilities will appear here.</p></div>', unsafe_allow_html=True)

    if classify:
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        elif not MODEL_PATH.exists():
            st.error(f"Model file not found: {MODEL_PATH.name}. Check that it exists in your GitHub repository.")
        else:
            try:
                with st.spinner("Analyzing image with the CNN..."):
                    model = load_model(str(MODEL_PATH))

                    # Match the model's input dimensions.
                    height, width = model.input_shape[1:3]
                    resized = image.resize((width, height))
                    array = np.asarray(resized, dtype=np.float32) / 255.0
                    array = np.expand_dims(array, axis=0)

                    scores = model.predict(array, verbose=0)[0]

                    if len(scores) != len(CLASS_NAMES):
                        st.error("The model output does not match the six configured classes. Verify the class order used during training.")
                    else:
                        best_index = int(np.argmax(scores))
                        confidence = float(scores[best_index]) * 100

                        st.success("Analysis complete")
                        st.markdown(
                            f'<div class="panel"><div style="color:#5eead4;font-size:.8rem;letter-spacing:2px">PREDICTED SCENE</div><h2 style="font-size:2rem;color:#f8fafc">{CLASS_NAMES[best_index].title()}</h2><div style="font-size:2rem;font-weight:800;color:#5eead4">{confidence:.1f}%</div><div style="color:#b8c7d8">Model confidence score</div></div>',
                            unsafe_allow_html=True
                        )

                        st.markdown("#### Class probability analysis")
                        for name, score in zip(CLASS_NAMES, scores):
                            st.write(f"**{name.title()}** · {float(score)*100:.1f}%")
                            st.progress(float(np.clip(score, 0, 1)))

                        st.caption("Confidence values are model scores, not a guarantee that the prediction is correct.")
            except Exception as error:
                st.error(f"Prediction failed: {error}")

st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#94a3b8;font-size:.8rem">TERRASCAN AI · Built with Streamlit & TensorFlow · Intel Image Classification</p>',
    unsafe_allow_html=True
)

