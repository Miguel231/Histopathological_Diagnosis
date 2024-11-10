import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from scipy import stats

# Autoencoder definition (same as in train_model.py)
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28 * 3, 128),  # Changed to 3 channels (RGB)
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
            nn.Linear(128, 28 * 28 * 3),  # Changed to 3 channels (RGB)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Custom dataset for loading patches
class PatchDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = self.annotations.iloc[idx]['path']
        img_path = os.path.join(self.img_dir, img_name)
        density = self.annotations.iloc[idx]['DENSITY']  # Label is now "DENSITY"
        
        # Threshold density to classify as healthy or abnormal
        # Define the threshold for density: if density > 0.5, label as "positive", else "negative"
        label = 'positive' if density > 0.5 else 'negative'

        image = Image.open(img_path).convert("RGB")  # Keep images in RGB
        if self.transform:
            image = self.transform(image)
        return image, label


# Function to load the model
def load_model(model, fold):
    model_filename = f"autoencoder_fold_{fold+1}.pth"
    model.load_state_dict(torch.load(model_filename))
    model.eval()  # Set the model to evaluation mode
    print(f"Model for fold {fold+1} loaded from {model_filename}")
    return model


# Function to compute reconstruction errors
def calculate_reconstruction_errors(model, data_loader):
    model.eval()
    errors = []
    with torch.no_grad():
        for img, _ in data_loader:
            img = img.view(img.size(0), -1)  # Flatten the image for the autoencoder
            output = model(img)
            error = torch.mean((output - img) ** 2, dim=1)
            errors.extend(error.cpu().numpy())
    return np.array(errors)


# Function to calculate adaptive threshold
def calculate_adaptive_threshold(errors, factor=1.5):
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return mean_error + factor * std_error


# Load CSV and initialize transforms
csv_file = "your_data.csv"  # Replace with your actual CSV path
img_dir = "your_image_folder"  # Replace with your actual image folder path
transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
dataset = PatchDataset(csv_file, img_dir, transform=transform)
patient_ids = dataset.annotations['patient_id'].values  # Ensure patient grouping for K-Fold

# Separate healthy (negative) and abnormal (positive) indices
healthy_indices = [i for i, label in enumerate(dataset.annotations['DENSITY']) if label <= 0.5]

# Set up validation set
val_subset = Subset(dataset, healthy_indices)  # You can change this to the full validation set
val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

# Load the model for a particular fold
fold = 0  # Specify which fold to load
autoencoder = AE()
load_model(autoencoder, fold)

# Calculate reconstruction errors on the validation set
print("Calculating reconstruction errors on validation set...")
val_errors = calculate_reconstruction_errors(autoencoder, val_loader)

# Calculate adaptive threshold
adaptive_threshold = calculate_adaptive_threshold(val_errors)
print(f"Adaptive threshold: {adaptive_threshold:.4f}")

# Anomaly detection on validation set
anomalies = []
true_labels = []  # Actual labels for computing metrics
predicted_labels = []  # Predicted labels for computing metrics

with torch.no_grad():
    for img, label in val_loader:
        img = img.view(img.size(0), -1)
        output = autoencoder(img)
        reconstruction_error = torch.mean((output - img) ** 2, dim=1).item()
        is_anomaly = reconstruction_error > adaptive_threshold
        anomalies.append((label, is_anomaly))
        true_labels.append(1 if label == "positive" else 0)  # Convert "positive" to 1, "negative" to 0
        predicted_labels.append(1 if is_anomaly else 0)

# Metrics calculations
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
cm = confusion_matrix(true_labels, predicted_labels)

# Confidence intervals for accuracy, precision, and recall
def compute_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin_of_error = z_score * (std_dev / np.sqrt(len(data)))
    return mean - margin_of_error, mean + margin_of_error

# Compute confidence intervals for accuracy, precision, and recall
accuracy_ci = compute_confidence_interval([accuracy])
precision_ci = compute_confidence_interval([precision])
recall_ci = compute_confidence_interval([recall])

print(f"Accuracy: {accuracy:.4f}, Confidence Interval: {accuracy_ci}")
print(f"Precision: {precision:.4f}, Confidence Interval: {precision_ci}")
print(f"Recall: {recall:.4f}, Confidence Interval: {recall_ci}")
print(f"Confusion Matrix:\n{cm}")

# Anomaly detection results
print("Anomaly detection results:")
for label, is_anomaly in anomalies:
    label_name = "Abnormal" if label == "positive" else "Healthy"
    status = "Anomaly" if is_anomaly else "Normal"
    print(f"Label: {label_name}, Detected: {status}")
