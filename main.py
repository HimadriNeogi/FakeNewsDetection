from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import requests

# Download model
MODEL_PATH = "saved_model"
MODEL_URL = "https://drive.google.com/uc?id=1bYp6DDRLQy9pDS3h0bnFun-VMr2Z4mm3"
MODEL_FILE = os.path.join(MODEL_PATH, "model.safetensors")

# Auto-download model if not present
if not os.path.exists(MODEL_FILE):
    print("Model not found locally. Downloading from Google Drive...")

    os.makedirs(MODEL_PATH, exist_ok=True)
    response = requests.get(MODEL_URL, stream=True)

    with open(MODEL_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Model download complete.")

# Initialize Flask app
app = Flask(__name__)

# Load model and tokenizer from saved_model/
MODEL_PATH = "saved_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Home route to serve HTML page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No input provided"}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty input"}), 400

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            label = "Real" if pred == 0 else "Fake"
            confidence = round(probs[0][pred].item() * 100, 2)

        return jsonify({"label": label, "confidence": confidence})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app locally for testing
if __name__ == "__main__":
    app.run(debug=True)