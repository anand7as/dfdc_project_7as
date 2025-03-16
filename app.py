from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Load DeepFake Detection Model
model = torch.load("deepfake_model.pth", map_location=torch.device("cpu"))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Route for homepage (renders the UI)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Preprocess the image
    image = Image.open(io.BytesIO(image.read()))
    image = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()

    return jsonify({"prediction": "Fake" if prediction > 0.5 else "Real", "confidence": round(prediction * 100, 2)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
