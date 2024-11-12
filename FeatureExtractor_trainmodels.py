import os
import pandas as pd
import PIL
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
from autoencoder import *
from classifier import CustomModel
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from itertools import product
from model_config import all_configurations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

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

def LoadAnnotated_1(df, data_dir):
    images = []
    labels = []
    
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        pat_id = row['CODI']  # Extract Pat_ID (patient identifier)
        label = row['DENSITAT']  # Extract DENSITY (patient diagnosis/label)
        
        for filename in os.listdir(data_dir):
            # Check if the filename starts with the Pat_ID (assuming the pattern 'Pat_ID_*')
            if filename.startswith(f"{pat_id}_"):
                file_path = os.path.join(data_dir, filename)
                
                # Check if the file exists
                if os.path.exists(file_path):
                    image = Image.open(file_path).convert("RGB")  # Open the image and convert to RGB
                    images.append(image)
                    labels.append(label)
                else:
                    print(f"Warning: File {file_path} not found.")
    
    return images, labels

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

def plot_loss_curve(epoch_losses, model_filename):
    # Plot the loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='b', label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve for {model_filename}")
    plt.legend()

    # Save the plot in the 'saved_models' folder
    save_path = os.path.join("saved_models", f"{model_filename}_loss_curve.png")
    os.makedirs("saved_models", exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved loss curve to {save_path}")
    plt.close()

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

# Track and save mean accuracy
model_accuracies = {}

model_decision = int(input("Select the method you want to proceed ( 0 = classifier and 1 = autoencoder): "))
if model_decision == 0:

    """
    #patient_data = pd.read_csv(r"C:/Users/larar/OneDrive/Documentos/Escritorio/Histopathological_Diagnosis-5/TRAIN_DATA_cropped.csv")
    #patient_data = patient_data.rename(columns={"CODI": "Pat_ID"})
    # Step 1: Load the Positive and Negative Patient Data
    #positive_patches = pd.read_csv("positive_patches.csv")
    #negative_patches = pd.read_csv("negative_patches.csv")
    # Concatenate to form a single DataFrame containing all patch-level data
    #all_annotations = pd.concat([positive_patches, negative_patches], ignore_index=True)
    # Set up binary Presence values for classification
    #all_annotations['Presence'] = all_annotations['Presence'].map({-1: 0, 1: 1})
    # Merge all_annotations with patient_data, adding `DENSITY` as a new column `patient_Diagnosis`
    #all_annotations = all_annotations.merge(patient_data[['Pat_ID', 'DENSITAT']], on="Pat_ID", how="left")
    # Rename `DENSITY` to `patient_Diagnosis`
    #all_annotations = all_annotations.rename(columns={"DENSITAT": "patient_Diagnosis"})
    # Save to a CSV file to inspect the merged data
    #all_annotations.to_csv("all_annotations_with_patient_diagnosis.csv", index=False)
    #print("Merged data with patient_Diagnosis column saved to 'all_annotations_with_patient_diagnosis.csv'.")
    
    """

    annotated_csv = pd.read_csv(r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis-5\all_annotations_with_patient_diagnosis.csv")

    # Load all images from `USABLE_annotated` using all_annotations
    data_dir = r"USABLE_annotated"
    img_list = LoadAnnotated(annotated_csv, data_dir)  # Assuming this function loads all images for the patches specified
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = StandardImageDataset(annotated_csv, img_list, transform=transform)

    print(f"Total number of configurations: {len(all_configurations)}")
    print("CREATE TRAIN AND TEST SUBSET WITH KFOLD")

    # Set up folder to save models
    save_folder = "saved_models"
    os.makedirs(save_folder, exist_ok=True)

    # Set up folder to save validations
    save_folder = "validation_data"
    os.makedirs(save_folder, exist_ok=True)

    # Group by `Pat_ID` and use the first `Presence` value for each patient as the stratification label
    patient_labels = annotated_csv.groupby('Pat_ID')['patient_Diagnosis'].first()

    # Set up Stratified K-Fold at patient level
    k_folds = 3
    strat_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
    fold_indices = []
    fold_accuracies = {}
    for fold, (train_patient_idx, val_patient_idx) in enumerate(strat_kfold.split(patient_labels.index, patient_labels)):
        print(f"Starting fold {fold + 1}/{k_folds}")
        
        # Select train and validation patient IDs
        train_patient_ids = patient_labels.index[train_patient_idx]
        val_patient_ids = patient_labels.index[val_patient_idx]
        
        # Filter patches based on these patient IDs
        train_df = annotated_csv[annotated_csv['Pat_ID'].isin(train_patient_ids)]
        val_df = annotated_csv[annotated_csv['Pat_ID'].isin(val_patient_ids)]
        
        # Collect indices for train and validation patches
        train_idx = train_df.index.tolist()
        val_idx = val_df.index.tolist()
        
        # Save fold indices
        fold_indices.append({'train': train_idx, 'val': val_idx})
        
        # Create subsets for training and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Data loaders
        train_loader = DataLoader(train_subset, batch_size=500, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=500, shuffle=False)

        # Save the validation indices (from 'val_patient_ids') and the associated dataset (e.g., validation annotations)
        val_subset_indices = {'val_patient_ids': val_patient_ids, 'val_indices': val_idx}
        torch.save(val_subset_indices, os.path.join(save_folder, f"val_subset_indices_fold{fold+1}.pth"))

        fold_accuracies[fold + 1] = []

        # Loop through each model configuration
        
        for config_idx, config in enumerate(all_configurations, start=1):
            print(f"Training config {config_idx}/{len(all_configurations)} in fold {fold + 1}")
            print(f"Configuration details: {config}")
            
            # Initialize the custom model
            custom_model = CustomModel(
                model_name=config["model_name"],
                embedding_dim=None,
                fc_layers=config["units_per_layer"],
                activations=["relu"] * config["num_layers"],
                batch_norms=[None] * config["num_layers"],
                dropout=config["dropout"],
                num_classes=2
            )
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            custom_model.to(device)
            
            # Define loss and optimizer
            pos_weight, neg_weight = weights(annotated_csv)  # Assumes this function calculates class weights
            weight = torch.tensor([pos_weight, neg_weight], device=device)
            criterion = nn.CrossEntropyLoss(weight=weight)
            optimizer = torch.optim.Adam(custom_model.parameters(), lr=0.01, weight_decay=1e-8)
            
            # Train the model
            epoch_losses, epoch_accuracies = train_model(custom_model, train_loader, criterion, optimizer, device, epochs=15)
            
            model_filename = (f"{config['model_name']}_{config['num_layers']}layers_"
                        f"{'_'.join(map(str, config['units_per_layer']))}_dropout{config['dropout']}_fold{fold + 1}")
            plot_loss_curve(epoch_losses, model_filename)

            # Calculate and save mean accuracy for this fold
            mean_fold_accuracy = np.mean(epoch_accuracies)
            fold_accuracies[fold + 1].append(mean_fold_accuracy)
            # Save the model after training
            filename = (f"{config['model_name']}_{config['num_layers']}layers_"
                        f"{'_'.join(map(str, config['units_per_layer']))}_dropout{config['dropout']}_fold{fold + 1}.pth")
            save_path = os.path.join(save_folder, filename)
            torch.save(custom_model.state_dict(), save_path)
            print(f"Saved model: {filename}")

elif model_decision == 1:
    annotations_file = pd.read_csv(r"TRAIN_DATA_cropped.csv")
    data_dir = r"C:\Users\larar\OneDrive\Documentos\Escritorio\USABLE_cropped"
    # Check if the directory exists
    if os.path.exists(data_dir):
        print(f"Directory found: {data_dir}")
        # Count the number of image files in the folder
        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_images = len(image_files)
        print(f"Number of images in the directory: {num_images}")
    else:
        print(f"Directory not found: {data_dir}")
    print("START LOAD FUNCTION")
    images_list, dict = LoadAnnotated_1(annotations_file, data_dir)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    print("START STANDARD DATASET")
    dataset = StandardImageDataset(annotations_file, images_list, transform=transform)
    print("LENGHT LIST IMAGES", len(images_list))
    print("FINISH STANDARD DATASET")
    print("CREATE TRAIN AND TEST SUBSET WITH KFOLD")
    fold_indices = []
    annotations_file['DENSITAT'] = annotations_file['DENSITAT'].map({-1: 0, 1: 1})  # map to binary labels
    grouped_annotations = annotations_file.groupby('CODI')
    grouped_labels = grouped_annotations['DENSITAT'].first()  # Use the first label in each CODI group for stratification
    k_folds = 3
    strat_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
    fold_indices = []
    thresholds = []
    fold_accuracies1 = []  
    # Use grouped CODI data to split into stratified folds
    for fold, (train_codi_idx, val_codi_idx) in enumerate(strat_kfold.split(grouped_labels.index, grouped_labels)):
        print(f"Starting fold {fold + 1}/{k_folds}")
        
        # Select train and validation CODI groups based on indices
        train_codi = grouped_labels.index[train_codi_idx]
        val_codi = grouped_labels.index[val_codi_idx]
        
        # Select patches in annotations_file where CODI is in the train or val CODI sets
        train_df = annotations_file[annotations_file['CODI'].isin(train_codi)]
        val_df = annotations_file[annotations_file['CODI'].isin(val_codi)]
        
        # Get index lists for train and validation patches
        train_idx = train_df.index.tolist()
        val_idx = val_df.index.tolist()

        # Save fold indices
        fold_indices.append({'train_auto': train_idx, 'val_auto': val_idx})
        
        # Create subsets for train and validation data
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Data loaders for train and validation subsets
        train_loader = DataLoader(train_subset, batch_size=500, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=500, shuffle=False)

        # Save the validation indices (from 'val_patient_ids') and the associated dataset (e.g., validation annotations)
        val_subset_indices = {'val_patient_ids': val_codi, 'val_indices': val_idx}
        torch.save(val_subset_indices, os.path.join("validation_data", f"val_subset_indices__autoencoder_fold{fold+1}.pth"))
        
        # Initialize the Autoencoder model
        model = AE()
        
        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Define loss and optimizer
        pos_weight, neg_weight = weights(annotations_file)
        weight = torch.tensor([pos_weight, neg_weight], device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train the autoencoder
        epoch_losses, epoch_accuracies = train_autoencoder(model, train_loader, criterion, optimizer, device, epochs=15)

        # Calculate mean accuracy for the fold
        mean_fold_accuracy = np.mean(epoch_accuracies)
        fold_accuracies1.append(mean_fold_accuracy)
        
        # Save the model after training
        filename = f"autoencoder_fold{fold + 1}.pth"
        save_path = os.path.join("saved_models", filename)
        torch.save(model.state_dict(), save_path)
        print(f"Saved autoencoder model for fold {fold + 1} at {save_path}")