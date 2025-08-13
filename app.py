import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN

# ------------------ CONSTANTS ------------------
RAF_DB_EMOTIONS = [
    "😲 Surprise",
    "😨 Fear",
    "🤢 Disgust",
    "😊 Happiness",
    "😢 Sadness",
    "😠 Anger",
    "😐 Neutral",
]

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# ------------------ LOAD FACE DETECTOR ------------------
@st.cache_resource
def load_face_detector():
    return MTCNN()

# ------------------ PREPROCESS IMAGE ------------------
def preprocess_image(image, target_size=(100, 100)):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=-1)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# ------------------ FACE DETECTION ------------------
def detect_face(image, detector):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 2:
        rgb_image = np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            rgb_image = image.copy()
        elif image.shape[2] == 4:
            rgb_image = image[:, :, :3]
        else:
            return None
    else:
        return None

    try:
        results = detector.detect_faces(rgb_image)
        if results:
            largest_face = max(results, key=lambda x: x["box"][2] * x["box"][3])
            x, y, w, h = largest_face["box"]
            padding = int(max(w, h) * 0.2)
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(rgb_image.shape[1], x + w + padding), min(rgb_image.shape[0], y + h + padding)
            return rgb_image[y1:y2, x1:x2]
        return None
    except Exception as e:
        st.error(f"❌ Face detection error: {e}")
        return None

# ------------------ PREDICT EMOTION ------------------
def predict_emotion(model, image):
    try:
        predictions = model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        return RAF_DB_EMOTIONS[predicted_class], confidence, predictions[0]
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return None, 0, None

# ------------------ STREAMLIT APP ------------------
st.set_page_config(page_title="Emotion Detection 🎭", page_icon="😊", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; color:#ff6600;'>🎭 Real-Time Emotion Detection</h1>
    <p style='text-align:center; font-size:18px; color:gray;'>
    Detect emotions from images with deep learning 🧠✨
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("⚙️ Model Configuration")
model_file = st.sidebar.file_uploader("📂 Upload .keras model", type=["keras"])

if model_file:
    with open("temp_model.keras", "wb") as f:
        f.write(model_file.getbuffer())

    try:
        st.sidebar.success("✅ Model loaded successfully!")
        model = load_model("temp_model.keras")

        if len(model.input_shape) == 4:
            target_h, target_w = model.input_shape[1], model.input_shape[2]
            target_size = (target_w, target_h)
        else:
            target_size = (100, 100)

        st.sidebar.info(f"📏 Target size: {target_size}")
        use_face_detection = st.sidebar.checkbox("🕵️ Use face detection", value=True)

        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["png", "jpg", "jpeg", "bmp", "tiff"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption="🖼️ Input Image", use_column_width=True)
            with col2:
                st.write("**ℹ️ Image Info:**")
                st.write(f"📏 Size: {image.size}")
                st.write(f"🎨 Mode: {image.mode}")
                if hasattr(image, "format"):
                    st.write(f"📄 Format: {image.format}")

            if st.button("🚀 Predict Emotion", type="primary", use_container_width=True):
                with st.spinner("⏳ Processing..."):
                    processed_image = None
                    if use_face_detection:
                        detector = load_face_detector()
                        face = detect_face(image, detector)
                        if face is not None:
                            st.success("😊 Face detected!")
                            st.image(face, caption="Detected Face", use_column_width=True)
                            processed_image = preprocess_image(face, target_size)
                        else:
                            st.warning("⚠️ No face detected, using full image")

                    if processed_image is None:
                        processed_image = preprocess_image(image, target_size)

                    emotion, confidence, all_preds = predict_emotion(model, processed_image)

                    if emotion is not None:
                        st.markdown(f"### 🎯 Predicted Emotion: **{emotion}**")
                        st.metric("🔥 Confidence", f"{confidence:.1%}")

                        st.subheader("📊 All Emotion Probabilities")
                        for emo, prob in zip(RAF_DB_EMOTIONS, all_preds):
                            st.progress(float(prob), text=f"{emo}: {prob:.3%}")
                    else:
                        st.error("❌ Prediction failed")

    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")

else:
    st.sidebar.warning("⚠️ Please upload a `.keras` model to begin")
    st.info("📥 Upload your trained emotion detection model in the sidebar to start")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>💡 Built with CNN + MTCNN | Powered by Streamlit</p>",
    unsafe_allow_html=True
)
