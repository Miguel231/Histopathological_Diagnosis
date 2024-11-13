import os
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from collections import OrderedDict
from itertools import product

# Set a custom cache directory for PyTorch models
#torch.hub.set_dir("C:/Users/larar/torch_cache")
torch.hub.set_dir(r"C:\Users\migue\torch_cache")

# Step 1: Generic function to load a model and return its feature size
def get_model_and_features(model_name, pretrained=True):
    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        num_features = model.fc.in_features
        model.fc = nn.Identity()  # Remove final classifier layer
    elif model_name == "densenet201":
        model = models.densenet201(weights="IMAGENET1K_V1" if pretrained else None)
        num_features = model.classifier.in_features
        model.classifier = nn.Identity()  # Remove final classifier layer
    else:
        raise ValueError(f"Model '{model_name}' is not supported. Choose 'resnet50' or 'densenet201'.")
    return model, num_features

# Step 2: Generic function to save model parameters
def save_model_parameters(model_name, folder, filename=None):
    # Define the filename based on the model name if not provided
    if filename is None:
        filename = f"{model_name}_params.npz"
    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    
    # Load the specified model
    model, _ = get_model_and_features(model_name)
    
    # Convert each parameter to numpy and save to .npz file
    params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(file_path, **params)
    
    print(f"Model parameters saved to {file_path}")

# Step 3: Generic function to load parameters into a model
def load_model_parameters(model_name, filename):
    model, num_features = get_model_and_features(model_name, pretrained=False)  # Initialize without pretrained weights
    params = np.load(filename)  # Load parameters from .npz file
    state_dict = model.state_dict()
    
    # Update model's state_dict with loaded parameters
    for k in state_dict.keys():
        if k in params:
            state_dict[k] = torch.tensor(params[k])  # Convert numpy array back to tensor
    model.load_state_dict(state_dict, strict=False)  # Load state dict with missing keys allowed
    
    return model, num_features

# Custom model class that supports both ResNet and DenseNet
class CustomModel(nn.Module):
    def __init__(self, model_name, embedding_dim, fc_layers, activations, batch_norms, dropout=0.25, pretrained_params=None, num_classes=2):
        super(CustomModel, self).__init__()
        
        # Load the specified model (ResNet or DenseNet) with optional pretrained parameters
        if pretrained_params:
            self.model, self.model_channels = load_model_parameters(model_name, pretrained_params)
        else:
            self.model, self.model_channels = get_model_and_features(model_name, pretrained=True)

        # Embedding layer setup
        if embedding_dim is not None:
            self.embedding = nn.Linear(self.model_channels, embedding_dim)  # If embedding is specified, reduce dimensionality
            self.in_features = embedding_dim
        else:
            self.embedding = None
            self.in_features = self.model_channels

        # Define fully connected layers
        self.fc_block = self.build_fc_block(self.in_features, fc_layers, activations, batch_norms, dropout)
        
        # Final classifier layer (output logits for classification)
        self.classifier = nn.Linear(fc_layers[-1], num_classes)  # Assuming the last layer in fc_layers is the output size

    def build_fc_block(self, in_features, fc_layers, activations, batch_norms, dropout):
        layers = OrderedDict()  # Sequential layer builder
        
        # Add layers to Sequential
        input_dim = in_features
        for i, output_dim in enumerate(fc_layers):
            # Fully connected layer
            layers[f'fc{i+1}'] = nn.Linear(input_dim, output_dim)
            
            # Activation function
            if activations[i] == 'relu':
                layers[f'relu{i+1}'] = nn.ReLU(inplace=True)
            elif activations[i] == 'tanh':
                layers[f'tanh{i+1}'] = nn.Tanh()
            elif activations[i] == 'relu6':
                layers[f'relu6{i+1}'] = nn.ReLU6(inplace=True)
            
            # Batch normalization if specified
            if batch_norms[i] == 'batch':
                layers[f'batchnorm{i+1}'] = nn.BatchNorm1d(output_dim)
            elif batch_norms[i] is None:
                pass  # No batch normalization

            # Dropout layer
            layers[f'dropout{i+1}'] = nn.Dropout(dropout)
            
            # Update input_dim for next layer
            input_dim = output_dim

        return nn.Sequential(layers)
    
    def forward(self, x):
        # Forward through model backbone (without classifier)
        x = self.model(x)
        x = torch.flatten(x, 1)  # Flatten the output
        
        # Forward through embedding layer if defined
        if self.embedding is not None:
            x = self.embedding(x)
        
        # Forward through FC block
        x = self.fc_block(x)
        
        # Forward through final classifier to get logits
        x = self.classifier(x)
        
        return x