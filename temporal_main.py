import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from autoencoder import AE
from classifier import * 

def LoadAnnotated(csv_path, data_dir):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize an empty list to store images
    IMS = []
    
    # Iterate over each row in the CSV
    for _, row in df.iterrows():
        # Construct the path to the image file
        pat_id = row['Pat_ID']
        window_id = str(row['Window_ID']).zfill(5)
        file_path = os.path.join(data_dir, f"{pat_id}_{window_id}.png")
        
        # Check if the file exists
        if os.path.exists(file_path):
            # Read the image and append to IMS (as a PIL object)
            image = Image.open(file_path).convert("RGB")  # Convert to RGB format if necessary
            IMS.append(image)
        else:
            print(f"Warning: File {file_path} not found.")
    
    return IMS



class StandardImageDataset(Dataset):
    def __init__(self, annotations_file, img_list, transform=None):
        """
        Args:
            annotations_file (str): Path to the Excel file with annotations.
            img_list (list): List of preloaded images in PIL format.
            transform (callable, optional): Optional transform to apply to the images.
        """
        # Load the annotations from the Excel file
        self.img_labels = pd.read_csv(annotations_file)
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        # Return the total number of samples
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get the image in PIL format from preloaded list
        image = self.img_list[idx]
        
        # Apply any image transformations (such as ToTensor, Resize)
        if self.transform:
            image = self.transform(image)  # Transform image as defined in the transform parameter
        
        # Convert image to float tensor with dtype=torch.float32
        if isinstance(image, torch.Tensor):
            image = image.float()
        
        label = self.img_labels.iloc[idx, 2]  # Label is in the third column
        label = 0 if label == -1 else label  # Convert -1 (if present) to 0 (negative class). as we are using crossentropyloss
        return image, label

def weights(annotated_file):
    df = pd.read_csv(annotated_file)
    
    # Initialize an empty list to store images
    positives = 0
    negatives =0
    c_general = 0
    # Iterate over each row in the CSV
    for _, row in df.iterrows():
        # Construct the path to the image file
        presence= row['Presence']
        if presence == -1:
            negatives+=1
        else:
            positives+=1
        c_general+=1

    return positives/c_general, negatives/c_general
def weights(annotated_file):
    df = pd.read_csv(annotated_file)
    
    # Initialize an empty list to store images
    positives = 0
    negatives = 0
    c_general = 0
    # Iterate over each row in the CSV
    for _, row in df.iterrows():
        # Construct the path to the image file
        presence = row['Presence']
        
        # Remap -1 to 0 (negatives) and 1 to positives
        if presence == -1:
            negatives += 1
        else:
            positives += 1
        c_general += 1

    return positives/c_general, negatives/c_general

       
annotations_file = r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\TRAIN_DATA.csv"
data_dir = r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\USABLE"

print("START LOAD FUNCTION")
# Load images as a list using LoadAnnotated
img_list = LoadAnnotated(annotations_file, data_dir)
print("FINISH LOAD FUNCTION")
# Define any transformations (e.g., resizing and converting to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),          # Convert images to tensor (C, H, W)
])
print("START STANDARD DATASET")
dataset = StandardImageDataset(annotations_file, img_list, transform=transform)
print("FINISH STANDARD DATASET")
# Use DataLoader for batching
print("START DATALOADER")
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("FINISH DATALOADER")


model_decision = 0
if model_decision == 1:
    print("INICIALIZE AUTOENCODER PROCESS")
    # Model Initialization
    model = AE()
else:
    print("INICIALIZE CLASSIFIER SECTION")
    folder=r'C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis'
    # Model Initialization
    # First, save DenseNet201 parameters
    save_densenet201_parameters(folder,"densenet201_params.npz")

    embedding_dim = 512  # Size of the embedding layer output, set None to skip embedding layer
    fc_layers = [256, 128]  # Sizes for each fully connected layer
    activations = ['relu', 'tanh']  # Activation functions for each FC layer
    batch_norms = ['batch', None]  # Batch normalization settings for each FC layer
    dropout = 0.25  # Dropout probability for each layer

    # Initialize the model
    list_models = CustomResNetModel(embedding_dim, fc_layers, activations, batch_norms, dropout, pretrained_params="densenet201_params.npz")
    model = list_models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loop over each fully connected layer configuration to save individually
    for i, layer_block in enumerate(model.fc_block):
        # Initialize a list to store layers
        layer_dict = OrderedDict()
        
        # Check if the layer_block is a Sequential object (which contains multiple layers)
        if isinstance(layer_block, nn.Sequential):
            for j, layer in enumerate(layer_block):
                # Ensure that the layer is of type nn.Linear before adding to the OrderedDict
                if isinstance(layer, nn.Linear):
                    layer_dict[f"linear_{i}_{j}"] = layer
                elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Tanh) or isinstance(layer, nn.ReLU6):
                    # Add activations if necessary
                    layer_dict[f"activation_{i}_{j}"] = layer
                elif isinstance(layer, nn.BatchNorm1d):
                    # Add batch normalization if necessary
                    layer_dict[f"batchnorm_{i}_{j}"] = layer
                elif isinstance(layer, nn.Dropout):
                    # Add dropout if necessary
                    layer_dict[f"dropout_{i}_{j}"] = layer
        else:
            # Handle the case where the layer_block is a single Linear layer
            if isinstance(layer_block, nn.Linear):
                layer_dict[f"linear_{i}_0"] = layer_block

        # Now we create the Sequential model with the filtered layers
        config_model = nn.Sequential(layer_dict)
        config_model.to(device)
        
        # Convert each parameter to numpy and save it with a unique filename
        params = {k: v.cpu().numpy() for k, v in config_model.state_dict().items()}
        np.savez(f"densenet201_params_config_{i}.npz", **params)
        
        print(f"Saved parameters for config {i} to 'densenet201_params_config_{i}.npz'")



pos_weight,neg_weight = weights(annotations_file)
weight = torch.tensor([pos_weight,neg_weight])
criterion = nn.CrossEntropyLoss(weight=weight)


# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1, weight_decay = 1e-8)

epochs = 5
outputs = []
losses = []
for epoch in range(epochs):
    epoch_loss = 0  # Initialize epoch loss
    i = 0
    for (image, label) in data_loader:
        image, label = image.to(device), label.to(device)  # Move to device (GPU/CPU)
        print(f"GO {i}")
        # Zero the gradients, forward pass, and calculate the loss
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(image)  # Model output shape: [batch_size, num_classes]

        # Calculate the loss: CrossEntropyLoss expects logits and class indices
        loss = criterion(logits, label)  # `label` should be the class indices

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()

        # Accumulate batch loss
        epoch_loss += loss.item()
        i+=1

    # Calculate the average loss over all batches in the epoch
    avg_epoch_loss = epoch_loss / len(data_loader)

    # Print the average loss for this epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")
    
