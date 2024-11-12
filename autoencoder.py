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
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (224, 224) -> (112, 112)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (112, 112) -> (56, 56)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (56, 56) -> (28, 28)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # (28, 28) -> (14, 14)
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # (14, 14) -> (7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 9)  # Bottleneck layer, compression
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(9, 1024 * 7 * 7),  # Reshape to the dimension before convolution
            nn.ReLU(),
            nn.Unflatten(1, (1024, 7, 7)),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # (7, 7) -> (14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # (14, 14) -> (28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (28, 28) -> (56, 56)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (56, 56) -> (112, 112)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (112, 112) -> (224, 224)
            nn.Sigmoid()  # Use Sigmoid to scale output between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, train_loader, criterion, optimizer, device, epochs=5):
    # Training loop for the autoencoder
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)  # No need to flatten anymore, the model handles it
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

