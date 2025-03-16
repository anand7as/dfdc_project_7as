from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
from torchvision import models

model = models.resnet18()  # Use the same model architecture you trained
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Adjust for binary classification
model.load_state_dict(torch.load("deepfake_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])



app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "DeepFake Detection API is live!"})

if __name__ == "__main__":
    app.run(debug=True)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read()))
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).item()

    return jsonify({"prediction": "Fake" if prediction > 0.5 else "Real", "confidence": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
