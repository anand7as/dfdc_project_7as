import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import io
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the model architecture (Replace this with your actual model)
import torch
import torch.nn as nn
import torchvision.models as models

class DeepFakeModel(nn.Module):
    def __init__(self):
        super(DeepFakeModel, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Ensure consistency with training
        num_ftrs = self.model.fc.in_features  # Get the input size of the last layer
        self.model.fc = nn.Linear(num_ftrs, 1)  # Change to 1 for binary classification, 2 for multi-class

    def forward(self, x):
        return self.model(x)


# Initialize the model
model = DeepFakeModel()
model.load_state_dict(torch.load("deepfake_model.pth", map_location=torch.device("cpu")), strict=False)
model.eval()


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
])

# API endpoint to predict deepfake
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.nn.functional.softmax(output, dim=1).max().item()

    result = {
        "prediction": "Fake" if prediction == 1 else "Real",
        "confidence": round(confidence * 100, 2)  # Convert to percentage
    }

    return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
