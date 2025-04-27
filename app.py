import os
import nltk
from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences

# If needed:
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# 1. Load Model & Tokenizer on Startup


# Load the .h5 model
model = tf.keras.models.load_model("merged_stress_model_01.h5")

# Load the tokenizer from .pkl
with open("mtokenizer01.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Adjust to match your training
max_length = 70
stop_words = set(stopwords.words('english'))



# 2. Preprocessing & Prediction Functions

def remove_stopwords(text):
    """
    Basic function to tokenize text and remove stopwords.
    Mirrors what you did during training.
    """
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if w.lower() not in stop_words]
    return " ".join(filtered_tokens)

def classify_stress(input_text):
    """
    Apply the same pipeline (remove_stopwords, tokenize, pad) and 
    then feed into the model to get 'stress' or 'no stress'.
    """
    # Clean the text
    cleaned_text = remove_stopwords(input_text)

    # Tokenize & pad
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')

    # Model predicts probabilities for 2 classes: [no_stress, stress]
    pred = model.predict(padded)
    label_idx = np.argmax(pred, axis=1)[0]  # 0 => no stress, 1 => stress
    return "stress" if label_idx == 1 else "no stress"



# 3. Flask Endpoint

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "text": "some text to classify" }
    Returns JSON: { "prediction": "stress" or "no stress" }
    """
    if not request.is_json:
        return jsonify({"error": "Request content-type must be application/json"}), 400

    data = request.get_json()
    user_text = data.get("text", "")

    if not user_text:
        return jsonify({"error": "No text field provided"}), 400

    # Classify text
    prediction_label = classify_stress(user_text)

    # Return JSON
    return jsonify({"prediction": prediction_label})




# # 4. Main entry point

# if __name__ == "__main__":
#     # Debug=True is handy during development. For production, consider removing debug mode.
#     app.run(host="0.0.0.0", port=5000, debug=True)


# ===== 4. Render-Specific Setup =====
port = int(os.environ.get("PORT", 5000))  # Render sets $PORT

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)  # debug=False in production
