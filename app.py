import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
import torchvision.models as models

app = Flask(__name__, static_folder="static", template_folder="templates")

# Define the model architecture
class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)  # Binary classification

    def forward(self, x):
        return self.model(x)

# Initialize and load the model
model = DeepFakeModel()
model.load_state_dict(torch.load("deepfake_model.pth", map_location=torch.device("cpu")), strict=False)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()

    result = {
        "prediction": "Fake" if prediction > 0.5 else "Real",
        "confidence": round(prediction * 100, 2)
    }

    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
