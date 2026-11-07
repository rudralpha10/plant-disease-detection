import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best_mobilenetv2.keras"
CLASSES_PATH = APP_DIR / "models" / "class_names.json"
LOG_DIR = APP_DIR / "logs"
LOG_PATH = LOG_DIR / "prediction_history.csv"
IMAGE_SIZE = (224, 224)

st.set_page_config(page_title="LeafGuard | Plant Disease AI", page_icon="🌿", layout="wide")

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

@st.cache_data
def load_classes(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        labels = data
    elif isinstance(data, dict):
        try:
            labels = [data[str(i)] for i in range(len(data))]
        except KeyError:
            labels = list(data.values())
    else:
        raise ValueError("class_names.json must contain a list or dictionary")
    if not labels:
        raise ValueError("No class names found in class_names.json")
    return [str(x) for x in labels]

def pretty_label(label):
    return label.replace("___", " — ").replace("_", " ")

def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)

def log_prediction(predicted_class, confidence, top_predictions):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["timestamp_utc", "predicted_class", "confidence", "top_predictions"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            predicted_class, f"{confidence:.6f}", json.dumps(top_predictions, ensure_ascii=False)
        ])

with st.sidebar:
    st.title("🌿 LeafGuard")
    st.caption("Plant disease classification")
    st.divider()
    st.write("Upload a clear photo of a single leaf to see the model's top predictions.")
    st.info("Educational demo only. Predictions may be unreliable on field photos or images unlike the training dataset.")

st.title("🌱 Plant Disease Classifier")
st.write("Analyze a leaf image with your trained **MobileNetV2** model.")

if not MODEL_PATH.exists():
    st.error(f"Model not found: `{MODEL_PATH}`. Put best_mobilenetv2.keras in the project's models folder.")
    st.stop()
if not CLASSES_PATH.exists():
    st.error(f"Class names file not found: `{CLASSES_PATH}`. Put class_names.json in the project's models folder.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    class_names = load_classes(str(CLASSES_PATH))
except Exception as exc:
    st.error("Could not load the model or class labels. Check that the model and matching class_names.json came from the same experiment.")
    st.exception(exc)
    st.stop()

if len(class_names) != int(model.output_shape[-1]):
    st.error(f"Class mismatch: JSON has {len(class_names)} labels but model outputs {model.output_shape[-1]} scores. Use the matching class_names.json.")
    st.stop()

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("1. Upload a leaf image")
    uploaded = st.file_uploader("Choose JPG, JPEG, or PNG", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
            st.image(image, caption="Uploaded leaf image", use_container_width=True)
        except Exception:
            st.error("Could not open this image. Please try another JPG or PNG.")
            st.stop()

with right:
    st.subheader("2. Prediction")
    if not uploaded:
        st.info("Upload an image to see predictions.")
    elif st.button("Analyze leaf", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                probs = np.asarray(model.predict(prepare_image(image), verbose=0))[0]
                if probs.ndim != 1 or len(probs) != len(class_names):
                    raise ValueError("Model output shape does not match class names.")
                probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                probs = np.clip(probs, 0, None)
                if probs.sum() <= 0:
                    raise ValueError("Model returned invalid scores.")
                probs = probs / probs.sum()
                indices = np.argsort(probs)[::-1][:3]
                top = [{"class": pretty_label(class_names[int(i)]), "confidence": float(probs[int(i)])} for i in indices]
                result = {"class": class_names[int(indices[0])], "confidence": float(probs[int(indices[0])]), "top": top}
                st.session_state["last_result"] = result
                log_prediction(result["class"], result["confidence"], top)
            except Exception as exc:
                st.error("Prediction failed.")
                st.exception(exc)

    result = st.session_state.get("last_result")
    if result:
        st.success("Analysis complete")
        st.markdown("**Predicted class**")
        st.subheader(pretty_label(result["class"]))
        st.metric("Model confidence", f"{result['confidence']:.1%}")
        if result["confidence"] < 0.60:
            st.warning("Confidence is relatively low. Try a clearer image and don't rely on this prediction alone.")
        st.markdown("**Top 3 predictions**")
        for item in result["top"]:
            st.write(f"{item['class']} — {item['confidence']:.1%}")
            st.progress(min(max(item["confidence"], 0.0), 1.0))

st.divider()
st.subheader("Prediction history")
st.caption("Logs timestamp, predicted class, confidence and top predictions. Uploaded images are not saved.")
if LOG_PATH.exists():
    try:
        import pandas as pd
        history = pd.read_csv(LOG_PATH)
        st.dataframe(history.tail(20), use_container_width=True, hide_index=True)
        st.download_button("Download prediction log (CSV)", LOG_PATH.read_bytes(), file_name="prediction_history.csv", mime="text/csv")
    except Exception as exc:
        st.warning(f"Could not display prediction history: {exc}")
else:
    st.write("No predictions logged yet.")

st.caption("Research and education demo — not a substitute for expert agricultural advice.")
