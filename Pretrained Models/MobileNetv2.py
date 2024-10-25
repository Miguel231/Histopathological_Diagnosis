import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torchvision import transforms
from PIL import Image

class FeatureExtractorMobileNet(nn.Module):
    def _init_(self, model_name, cut_index=None):
        super()._init_()
        # Load the pre-trained DenseNet201 model
        self.fe = models.MobileNetV2(pretrained=True)
        
        # Remove the final classification layer (i.e. the fully connected layer)
        self.fe.classifier = nn.Identity()
        
    def forward(self, x):
        x = self.fe(x)
        return x

def preprocess_image(image_path): # TENIM MATEIX FUNCIÓ LARA I MERI
    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to MobileNetV2's input size
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])

# Run model once and save features
def save_features(image_paths, output_file):
    model = FeatureExtractorMobileNet()  # Instantiate the model
    model.eval()  # Set to evaluation mode
    
    features = []
    with torch.no_grad():  # Disable gradient calculation for faster inference
        for image_path in image_paths:
            image = preprocess_image(image_path)
            output = model(image)  # Get features
            features.append(output.squeeze(0).numpy())  # Save features as numpy array
    
    # Save extracted features to npz file
    np.savez(output_file, *features)
    print(f"Features saved to {output_file}")