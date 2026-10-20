"""Simple Streamlit interface for the trained SMS spam model."""
from pathlib import Path
import json
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "sms_spam_lstm.keras"
TOKENIZER_PATH = ROOT / "artifacts" / "tokenizer.json"
CONFIG_PATH = ROOT / "artifacts" / "config.json"

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")
st.title("📩 SMS and Message Spam Detection")
st.write("Enter a message to classify it as **Spam** or **Not Spam**.")

if not (MODEL_PATH.exists() and TOKENIZER_PATH.exists() and CONFIG_PATH.exists()):
    st.warning("The model files are not available yet. Run `python train_model.py` first, then restart this app.")
    st.code("python train_model.py\nstreamlit run app.py", language="bash")
    st.stop()

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    return model, tokenizer, config

model, tokenizer, config = load_assets()
message = st.text_area("Type your SMS message:", placeholder="Example: Hi, are you coming to class tomorrow?", height=130)

if st.button("Predict", type="primary", use_container_width=True):
    if not message.strip():
        st.info("Please enter a message first.")
    else:
        sequence = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(sequence, maxlen=config["max_len"], padding="post", truncating="post")
        probability = float(model.predict(padded, verbose=0)[0][0])
        if probability >= config["threshold"]:
            st.error(f"Prediction: SPAM (model score: {probability:.1%})")
        else:
            st.success(f"Prediction: NOT SPAM (spam score: {probability:.1%})")
        st.caption("This is an educational prediction and may be wrong. Do not rely on it to decide whether a link or sender is safe.")

with st.expander("Try an example"):
    st.write("Not spam: Hi, are you coming to college tomorrow?")
    st.write("Possible spam: Congratulations! You have won a prize. Claim it now!")
