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


def LoadAnnotated(df, data_dir):
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
        self.img_labels = annotations_file
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
    df = annotated_file
    
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
    df = annotated_file
    
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

       
#annotations_file = r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\TRAIN_DATA.csv"
annotations_file = r"TRAIN_DATA.csv"


#data_dir = r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\USABLE"
data_dir = r"USABLE"

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
data_loader = DataLoader(dataset, batch_size=500, shuffle=True)
print("FINISH DATALOADER")


model_decision = 0
if model_decision == 1:
    print("INICIALIZE AUTOENCODER PROCESS")
    # Model Initialization
    model = AE()
else:
    print("INICIALIZE CLASSIFIER SECTION")
    #folder=r'C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis'
    folder=r'/export/fhome/vlia04/MyVirtualEnv/Histopathological_Diagnosis/'

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
model_decision = int(input("Select the method you want to proceed ( 0 = classifier and 1 = autoencoder): "))
if model_decision == 0:
    annotations_file = pd.read_csv(r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\TRAIN_DATA.csv")
    data_dir = r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\USABLE"
    patient_groups = annotations_file.groupby('Pat_ID')
    print("START LOAD FUNCTION")
    # Load images as a list using LoadAnnotated
    img_list = LoadAnnotated(annotations_file, data_dir)
    print("FINISH LOAD FUNCTION")
    # Define transformations
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    print("START STANDARD DATASET")
    dataset = StandardImageDataset(annotations_file, img_list, transform=transform)
    print("FINISH STANDARD DATASET")
    print("CREATE TRAIN AND TEST SUBSET WITH KFOLD")
    # Prepare cross-validation split
    k_folds = 5
    unique_patient_ids = annotations_file['Pat_ID'].unique()
    patient_labels = annotations_file.groupby('Pat_ID')['Presence'].max().map({-1: 0, 1: 1})  # Binary labels

    # Stratified K-Fold on unique patients
    strat_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
    all_configurations = []

    # Define ranges for configurations
    model_options = ["resnet50", "densenet201"]  # Model architectures
    num_layers_options = range(1, 4)  # Number of layers (1 to 3)
    units_per_layer_options = [64, 128, 256]  # Units per layer
    dropout_options = [0.25]  # Dropout probabilities

    # Folder to save models
    save_folder = "saved_models"
    os.makedirs(save_folder, exist_ok=True)
    # Generate all possible configurations (for layers with different unit values)
    for num_layers in num_layers_options:
        # Get layer configurations, filtering out unwanted combinations
        layer_configs = list(product(units_per_layer_options, repeat=num_layers))
        # Filter out configurations where layers have repeated values based on num_layers
        if num_layers == 2:
            layer_configs = [config for config in layer_configs if len(set(config)) == 2]
        elif num_layers == 3:
            layer_configs = [config for config in layer_configs if len(set(config)) > 1]
        # Now combine with model names and dropout options
        for model_name, layer_config, dropout in product(model_options, layer_configs, dropout_options):
            configuration = {
                "model_name": model_name,
                "num_layers": num_layers,
                "units_per_layer": list(layer_config),
                "dropout": dropout
            }
            all_configurations.append(configuration)

    # Print total number of configurations and details
    print(f"Total number of configurations: {len(all_configurations)}")
    # Initialize fold_indices to save the indices of train and validation data for each fold
    fold_indices = []

    # Train models for each fold and configuration
    for fold, (train_patient_idx, val_patient_idx) in enumerate(strat_kfold.split(unique_patient_ids, patient_labels)):
        print(f"\nStarting fold {fold + 1}/{k_folds}")

        # Get patient IDs for train and validation
        train_patients = unique_patient_ids[train_patient_idx]
        val_patients = unique_patient_ids[val_patient_idx]

        # Indices for patches belonging to train and validation patients
        train_df = annotations_file[annotations_file['Pat_ID'].isin(train_patients)]
        val_df = annotations_file[annotations_file['Pat_ID'].isin(val_patients)]
        train_idx = train_df.index.tolist()
        val_idx = val_df.index.tolist()

        # Create subset datasets for training and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Define data loaders for training and validation
        train_loader = DataLoader(train_subset, batch_size=500, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=500, shuffle=False)

        # Loop through each configuration
        for config_idx, config in enumerate(all_configurations, start=1):
            print(f"Training config {config_idx}/{len(all_configurations)} in fold {fold + 1}")

            # Initialize the model for each configuration
            custom_model = CustomModel(
                model_name=config["model_name"],
                embedding_dim=None,  # Set to None, or change as needed
                fc_layers=config["units_per_layer"],
                activations=["relu"] * config["num_layers"],
                batch_norms=[None] * config["num_layers"],  # Adjust batch norms if needed
                dropout=config["dropout"],
                num_classes=2  # Assuming binary classification
            )
            
            # Define device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            custom_model.to(device)

            # Define loss and optimizer
            pos_weight, neg_weight = weights(annotations_file)
            weight = torch.tensor([pos_weight, neg_weight], device=device)
            criterion = nn.CrossEntropyLoss(weight=weight)
            optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.01, weight_decay=1e-8)


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
        #print(f"GO {i}")
        # Zero the gradients, forward pass, and calculate the loss
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(image)  # Model output shape: [batch_size, num_classes]

        # Calculate the loss: CrossEntropyLoss expects logits and class indices
        loss = criterion(logits, label)  # `label` should be the class indices

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()
            # Training loop
            epochs = 5
            for epoch in range(epochs):
                custom_model.train()
                train_loss, correct_train, total_train = 0.0, 0, 0

                for images, labels in train_loader:
                    print("HELLO")
                    images, labels = images.to(device), labels.to(device)
                    
                    # Forward pass and loss calculation
                    optimizer.zero_grad()
                    logits = custom_model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(logits, 1)
                    correct_train += (predicted == labels).sum().item()
                    total_train += labels.size(0)

                avg_train_loss = train_loss / total_train
                train_accuracy = correct_train / total_train * 100
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

            # Validation loop
            custom_model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = custom_model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
                    correct_val += (predicted == labels).sum().item()
                    total_val += labels.size(0)

            avg_val_loss = val_loss / total_val
            val_accuracy = correct_val / total_val * 100
            print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

            # Save model and configuration
            filename = (f"{config['model_name']}_{config['num_layers']}layers_"
                        f"{'_'.join(map(str, config['units_per_layer']))}_dropout{config['dropout']}_fold{fold + 1}.pth")
            save_path = os.path.join(save_folder, filename)
            torch.save(custom_model.state_dict(), save_path)
            print(f"Saved configuration: {filename}")

    # Save fold indices
    np.savez('saved_fold_indices.npz', fold_indices=[{'train': train_idx, 'val': val_idx} for train_idx, val_idx in fold_indices])
if model_decision == 1:
    print("AUTOENCODER")