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

def train_model(custom_model, train_loader, criterion, optimizer, device, epochs=5):
    # To store the loss and accuracy for each epoch
    epoch_losses = []
    epoch_accuracies = []

    # Training loop
    for epoch in range(epochs):
        custom_model.train()  # Set the model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = custom_model(images)

            # Compute the loss
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Track the training loss
            train_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        # Average loss and accuracy for the epoch
        avg_train_loss = train_loss / total_train
        train_accuracy = correct_train / total_train * 100

        # Store the loss and accuracy for the current epoch
        epoch_losses.append(avg_train_loss)
        epoch_accuracies.append(train_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
    
    return epoch_losses, epoch_accuracies

model_decision = int(input("Select the method you want to proceed ( 0 = classifier and 1 = autoencoder): "))
if model_decision == 0:
    #annotations_file = pd.read_csv(r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\TRAIN_DATA.csv")
    annotations_file = pd.read_csv(r"TRAIN_DATA.csv")

    #data_dir = r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis\USABLE"
    data_dir = r"USABLE_annotated"

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
    
    # Define ranges for configurations
    model_options = ["resnet50", "densenet201"]  # Model architectures
    num_layers_options = range(1, 4)  # Number of layers (1 to 3)
    units_per_layer_options = [64, 128, 256]  # Units per layer
    dropout_options = [0.25]  # Dropout probabilities
    all_configurations = []
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
    print("CREATE TRAIN AND TEST SUBSET WITH KFOLD")
    # Initialize fold_indices to save the indices of train and validation data for each fold
    fold_indices = []
    annotations_file['Presence'] = annotations_file['Presence'].map({-1: 0, 1: 1})  # map to binary labels
    # 1. Define your patch labels
    patch_labels = annotations_file['Presence'].values
    # 2. Perform Stratified K-Fold split based on patch labels (Presence)
    k_folds = 5
    strat_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    # 3. Initialize fold_indices to save indices for train and validation sets
    fold_indices = []

    # Perform Stratified K-Fold split based on patch-level labels
    for fold, (train_idx, val_idx) in enumerate(strat_kfold.split(annotations_file, patch_labels)):
        print(f"Starting fold {fold + 1}/{k_folds}")
        # Collect patch indices for training and validation sets
        train_df = annotations_file.iloc[train_idx]
        val_df = annotations_file.iloc[val_idx]

        # Save indices to fold_indices
        fold_indices.append({'train': train_idx, 'val': val_idx})

        # Now create subsets for training and validation based on these indices
        train_subset = Subset(dataset, train_idx)  # Use your dataset object
        val_subset = Subset(dataset, val_idx)  # Use your dataset object

        # Define dataloaders for training and validation sets
        train_loader = DataLoader(train_subset, batch_size=500, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=500, shuffle=False)

        # Loop through each configuration
        for config_idx, config in enumerate(all_configurations, start=1):
            print(f"Training config {config_idx}/{len(all_configurations)} in fold {fold + 1}")
            print(f"Configuration details: {config}")  
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
            criterion = nn.CrossEntropyLoss(weight=weight) #expect integers
            optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.01, weight_decay=1e-8)

            # Train the model and get the loss/accuracy per epoch
            epoch_losses, epoch_accuracies = train_model(custom_model, train_loader, criterion, optimizer, device, epochs=15)

            # Save the model weights after training
            filename = (f"{config['model_name']}_{config['num_layers']}layers_"
                        f"{'_'.join(map(str, config['units_per_layer']))}_dropout{config['dropout']}_fold{fold + 1}.pth")
            save_path = os.path.join(save_folder, filename)
            torch.save(custom_model.state_dict(), save_path)
            print(f"Saved model: {filename}")

elif model_decision == 1:
    annotations_file = pd.read_csv(r"TRAIN_DATA_cropped.csv")
    data_dir = r"USABLE_annotated"
    patient_groups = annotations_file.groupby('Pat_ID')
    print("START LOAD FUNCTION")
    
    # Load images using the LoadAnnotated function
    img_list = LoadAnnotated(annotations_file, data_dir)
    print("FINISH LOAD FUNCTION")
    
    # Define image transformations (resize to 224x224 and convert to tensor)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    print("START STANDARD DATASET")
    
    dataset = StandardImageDataset(annotations_file, img_list, transform=transform)
    print("FINISH STANDARD DATASET")
    
    # Stratified K-Fold split
    print("CREATE TRAIN AND TEST SUBSET WITH KFOLD")
    fold_indices = []
    annotations_file['Presence'] = annotations_file['Presence'].map({-1: 0, 1: 1})  # map to binary labels
    patch_labels = annotations_file['Presence'].values
    k_folds = 5
    strat_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    # Initialize fold_indices to save indices for each fold
    for fold, (train_idx, val_idx) in enumerate(strat_kfold.split(annotations_file, patch_labels)):
        print(f"Starting fold {fold + 1}/{k_folds}")
        train_df = annotations_file.iloc[train_idx]
        val_df = annotations_file.iloc[val_idx]
        
        # Create subsets for train and validation data
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=500, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=500, shuffle=False)
        
        # Initialize the Autoencoder model
        model = AE()
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Define loss and optimizer
        pos_weight, neg_weight = weights(annotations_file)
        weight = torch.tensor([pos_weight, neg_weight], device=device)
        criterion = nn.CrossEntropyLoss(weight=weight) #expect integers
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train the autoencoder
        train_autoencoder(model, train_loader, criterion, optimizer, device, epochs=15)
        
        # Calculate reconstruction errors on the validation set
        val_errors = calculate_reconstruction_errors(model, val_loader, device)
        
        # Calculate adaptive threshold for anomaly detection
        adaptive_threshold = calculate_adaptive_threshold(val_errors)
        print(f"Adaptive threshold for anomaly detection: {adaptive_threshold:.4f}")
        
        # Save the model after training
        filename = f"autoencoder_fold{fold + 1}.pth"
        save_path = os.path.join("saved_models", filename)
        torch.save(model.state_dict(), save_path)
        print(f"Saved autoencoder model for fold {fold + 1} at {save_path}")