import os
import torch

# Set a custom cache directory
torch.hub.set_dir("C:/Users/larar/torch_cache")

import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from collections import OrderedDict
from torchvision.models import densenet201, DenseNet201_Weights

def save_densenet201_parameters(folder, filename="densenet201_params.npz"):
    # Ensure the folder exists; if not, create it
    os.makedirs(folder, exist_ok=True)
    # Define the full path including the folder and filename
    file_path = os.path.join(folder, filename)
    
    # Load DenseNet201 model with explicit weights
    weights = DenseNet201_Weights.IMAGENET1K_V1
    model = densenet201(weights=weights)
    
    # Convert each parameter to numpy and save to .npz file
    params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(file_path, **params)
    
    print(f"Model parameters saved to {file_path}")

# Step 2: Function to Load Parameters from NPZ File into DenseNet201 Model
def load_densenet201_parameters(filename="densenet201_params.npz"):
    model = models.densenet201(pretrained=False)  # Initialize DenseNet201 without pretrained weights
    params = np.load(filename)  # Load parameters from npz file
    state_dict = model.state_dict()
    
    # Update model's state_dict with loaded parameters
    for k in state_dict.keys():
        if k in params:
            state_dict[k] = torch.tensor(params[k])  # Convert numpy array back to tensor
    model.load_state_dict(state_dict, strict=False)  # Load state dict with missing keys allowed

    # Remove the original classifier for custom fully connected layers
    num_features = model.classifier.in_features
    model.classifier = nn.Identity()  # Remove final classifier layer for custom configuration
    
    return model, num_features

class CustomResNetModel(nn.Module):
    def __init__(self, embedding_dim, fc_layers, activations, batch_norms, dropout=0.25, pretrained_params=None, num_classes=2):
        super(CustomResNetModel, self).__init__()

        # Load DenseNet201 with pretrained parameters if specified
        if pretrained_params:
            self.model, self.densenet_channels = load_densenet201_parameters(pretrained_params)
        else:
            self.model = models.densenet201(pretrained=True)
            self.densenet_channels = self.model.classifier.in_features
            self.model.classifier = nn.Identity()

        # Embedding layer setup
        if embedding_dim is not None:
            self.embedding = nn.Linear(self.densenet_channels, embedding_dim)
            self.in_features = embedding_dim
        else:
            self.embedding = None
            self.in_features = self.densenet_channels

        # Define fully connected layers
        self.fc_block = self.build_fc_block(self.in_features, fc_layers, activations, batch_norms, dropout)

        # Final classifier layer (output logits for classification)
        self.classifier = nn.Linear(fc_layers[-1], num_classes)

    def forward(self, x):
        # Forward through DenseNet backbone (without classifier)
        x = self.model(x)
        x = torch.flatten(x, 1)

        # Forward through embedding layer if defined
        if self.embedding is not None:
            x = self.embedding(x)

        # Forward through FC block
        x = self.fc_block(x)

        # Forward through final classifier to get logits
        x = self.classifier(x)

        return x
