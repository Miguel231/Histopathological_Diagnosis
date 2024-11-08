import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# Step 1: Function to Save DenseNet201 Pretrained Model Parameters
def save_densenet201_parameters(filename="densenet201_params.npz"):
    model = models.densenet201(pretrained=True)  # Load pretrained DenseNet201 model
    params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}  # Convert each parameter to numpy
    np.savez(filename, **params)  # Save all parameters to .npz file

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

# Step 3: Build Custom DenseNet201 Model with Fully Connected Layers
def build_custom_densenet201(config, pretrained_params=None):
    # Load DenseNet201 with pretrained parameters if specified
    if pretrained_params:
        model, num_features = load_densenet201_parameters(pretrained_params)
    else:
        model = models.densenet201(pretrained=False)
        num_features = model.classifier.in_features
        model.classifier = nn.Identity()
    
    # Custom fully connected layers
    layers = []
    activation_fn = nn.ReLU() if config['activation'] == 'relu' else nn.Tanh() if config['activation'] == 'tanh' else nn.LeakyReLU(0.01)
    
    for units in config['num_units']:
        layers.append(nn.Linear(num_features, units))
        layers.append(activation_fn)
        layers.append(nn.Dropout(0.5))
        num_features = units

    layers.append(nn.Linear(num_features, 1))
    if config['loss_fn'] == 'bce_logits':
        layers.append(nn.Identity())
    else:
        layers.append(nn.Sigmoid())

    model.classifier = nn.Sequential(*layers)
    
    return model

# Step 4: Configuration and Usage
config = {
    'activation': 'relu',
    'num_units': [1024, 512],
    'loss_fn': 'bce',
    'optimizer': 'adam',
    'learning_rate': 0.001
}


# Then, load DenseNet201 model with the saved parameters and build custom fully connected layers
model = build_custom_densenet201(config, pretrained_params="densenet201_params.npz")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)