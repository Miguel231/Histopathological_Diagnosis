import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from FeatureExtractor_trainmodels import *
import seaborn as sns

# Function to load the model from a .pth file
def load_model(model_path, device,config):
    model = CustomModel(
                model_name=config["model_name"],
                embedding_dim=None,
                fc_layers=config["units_per_layer"],
                activations=["relu"] * config["num_layers"],
                batch_norms=[None] * config["num_layers"],
                dropout=config["dropout"],
                num_classes=2
            )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to evaluate a model on the dataset
def evaluate_model(model, val_loader, device):
    model.eval()  # Ensure model is in evaluation mode
    all_labels = []
    all_preds = []
    patient_predictions = {}
    patient_labels = {}

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, precision, recall, f1, conf_matrix

def plot_confusion_matrix(conf_matrix, save_path=None):
    plt.figure(figsize=(8, 6))
    # Display the confusion matrix with annotations
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path)
    plt.show()


# Function to save evaluation results
def save_evaluation_results(evaluation_results, save_folder, model_filename):
    eval_file = os.path.join(save_folder, f"{model_filename}_evaluation_results.csv")
    eval_df = pd.DataFrame(evaluation_results)
    eval_df.to_csv(eval_file, index=False)
    print(f"Saved evaluation results to {eval_file}")

# Main evaluation loop
def evaluate_models(saved_models_folder, val_loader, device):
    evaluation_metrics = []

    for model_filename in os.listdir(saved_models_folder):
        if model_filename.endswith(".pth"):
            print(f"Evaluating model: {model_filename}")
            
            # Load the saved model
            model_path = os.path.join(saved_models_folder, model_filename)
            config = {
                    "model_name": "densenet201",
                    "num_layers": 1,
                    "units_per_layer": [64],
                    "dropout": 0.25
                }
            model = load_model(model_path, device,config)
            
            # Evaluate the model
            accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, val_loader, device)
            
            # Save evaluation metrics
            evaluation_results = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": [conf_matrix.tolist()]  # Convert to list for saving
            }
            evaluation_metrics.append(evaluation_results)
            
            # Plot and save confusion matrix
            cm_save_path = os.path.join(saved_models_folder, f"{model_filename}_confusion_matrix.png")
            plot_confusion_matrix(conf_matrix, cm_save_path)

    # Save all evaluation results to CSV
    evaluation_df = pd.DataFrame(evaluation_metrics)
    evaluation_df_file = os.path.join(saved_models_folder, "all_evaluation_results.csv")
    evaluation_df.to_csv(evaluation_df_file, index=False)
    print(f"Saved all evaluation results to {evaluation_df_file}")

mode = input("Do you want to validate (0 = classifier and 1 = autoencoder)?: ")

if mode == "0":
    annotated_csv = pd.read_csv(r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis-5\all_annotations_with_patient_diagnosis.csv")
    # Load all images from `USABLE_annotated` using all_annotations
    data_dir = r"USABLE_annotated"
    img_list = LoadAnnotated(annotated_csv, data_dir)  # Assuming this function loads all images for the patches specified
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = StandardImageDataset(annotated_csv, img_list, transform=transform)
    # Set up folder to save models
    save_folder = "save_evaluations"
    os.makedirs(save_folder, exist_ok=True)
    # You can now filter the dataset again based on these indices if needed
    val_subset_indices = torch.load(r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis-5\validation_data\val_subset_indices_fold1.pth")
    val_subset = Subset(dataset, val_subset_indices['val_indices'])
    # Create the validation DataLoader using the same batch size and shuffle settings
    val_loader = DataLoader(val_subset, batch_size=500, shuffle=False)
    # Define the device (CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Folder containing the saved models
    saved_models_folder = "save_evaluations"
    # Start evaluating all models
    evaluate_models(saved_models_folder, val_loader, device)