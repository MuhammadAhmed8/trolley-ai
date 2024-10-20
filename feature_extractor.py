import torch
from PIL import Image
import numpy as np

class FeatureExtractor:
    def __init__(self, model_path='models/resnet50.pth'):
        # Load a pre-trained ResNet model
        self.model = torch.load(model_path)
        self.model.eval()

    def extract_features(self, image):
        # Preprocess image
        image = image.resize((224, 224))  # Resize for ResNet
        image = np.array(image) / 255.0
        image = np.transpose(image, (2, 0, 1))  # Change to CHW format
        image = torch.tensor(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            features = self.model(image)  # Extract features
        return features.numpy()
