from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

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
        text = data.get("text", "")
        if not text.strip():
            return jsonify({"error": "Empty input"}), 400

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            label = "Real" if pred == 0 else "Fake"
            confidence = round(probs[0][pred].item() * 100, 2)

        return jsonify({ "label": label, "confidence": confidence })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app locally for testing
if __name__ == "__main__":
    app.run(debug=True)