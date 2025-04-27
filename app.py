import os
import nltk
from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np

# Memory optimization for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# NLTK persistent setup
nltk_data_path = '/tmp/nltk_data'
os.makedirs(nltk_data_path, exist_ok=True)
required_nltk = ['stopwords', 'punkt', 'punkt_tab']  # Added 'punkt_tab'
for package in required_nltk:
    try:
        nltk.data.find(f'tokenizers/{package}')
    except LookupError:
        nltk.download(package, download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Flask app setup
app = Flask(__name__)

# Globals for lazy loading
model = None
tokenizer = None

# Model parameters
max_length = 70
stop_words = set(stopwords.words('english'))


# Utility functions

def load_model_and_tokenizer():
    """Lazy-load model and tokenizer when first needed."""
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Loading model and tokenizer...")
        model = tf.keras.models.load_model("merged_stress_model_01.h5")
        with open("mtokenizer01.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        print("Model and tokenizer loaded successfully.")

def remove_stopwords(text):
    """Tokenize text and remove stopwords."""
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
    return " ".join(filtered_tokens)

def classify_stress(input_text):
    """Preprocess input and predict stress."""
    cleaned_text = remove_stopwords(input_text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    pred = model.predict(padded)
    label_idx = np.argmax(pred, axis=1)[0]
    return "stress" if label_idx == 1 else "no stress"


# Routes

@app.route("/", methods=["GET"])
def home():
    """Home route for health check."""
    return "Server is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    """Predict stress from input text."""
    if not request.is_json:
        return jsonify({"error": "Request content-type must be application/json"}), 400

    data = request.get_json()
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No text field provided"}), 400

    load_model_and_tokenizer()  # Lazy load model & tokenizer
    prediction_label = classify_stress(user_text)

    return jsonify({"prediction": prediction_label})


# Main entry point for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
