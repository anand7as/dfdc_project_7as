import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from flask import Flask, request, jsonify
import torchvision.models as models

app = Flask(__name__)

# Define the model architecture
class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")  # Updated from pretrained=True
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

# ✅ Root endpoint to check if app is running
@app.route('/')
def home():
    return jsonify({"message": "DeepFake Detection API is running!"})

# ✅ Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()  # Use sigmoid for binary classification

    result = {
        "prediction": "Fake" if prediction > 0.5 else "Real",
        "confidence": round(prediction * 100, 2)  # Convert to percentage
    }

    return jsonify(result)

# ✅ Run the app with Render's dynamic port
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's port
    app.run(host="0.0.0.0", port=port)
