import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from autoencoder import *
from classifier import *
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from itertools import product

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(224 * 224 * 3, 128),  # Assuming images are resized to 224x224 and RGB
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 9)
        )
        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 224 * 224 * 3),  # Assuming output is 224x224 RGB
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def calculate_reconstruction_errors(model, data_loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.view(images.size(0), -1).to(device)  # Flatten the image for the autoencoder
            output = model(images)
            error = torch.mean((output - images) ** 2, dim=1)
            errors.extend(error.cpu().numpy())
    return np.array(errors)

def calculate_adaptive_threshold(errors, factor=1.5):
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return mean_error + factor * std_error

def train_autoencoder(model, train_loader, criterion, optimizer, device, epochs=5):
    # Training loop for the autoencoder
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.view(images.size(0), -1).to(device)  # Flatten the image
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
