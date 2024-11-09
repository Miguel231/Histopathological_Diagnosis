import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from FeatureExtractor_trainmodels import *
from classifier import * 

# Function to load the model from a .pth file
def load_model(model_path, custom_model_class, device):
    model = custom_model_class()  # Initialize your custom model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to evaluate a model on the dataset
def evaluate_model(model, val_loader, device):
    model.eval()  # Ensure model is in evaluation mode
    all_labels = []
    all_preds = []
    
    with torch.no_grad():  # Disable gradient computation for inference
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

# Function to plot confusion matrix
def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    tick_marks = range(2)  # Assuming binary classification (change if more classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(tick_marks)
    ax.set_yticklabels(tick_marks)
    
    # Add confusion matrix values
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to save evaluation results
def save_evaluation_results(evaluation_results, save_folder, model_filename):
    eval_file = os.path.join(save_folder, f"{model_filename}_evaluation_results.csv")
    eval_df = pd.DataFrame(evaluation_results)
    eval_df.to_csv(eval_file, index=False)
    print(f"Saved evaluation results to {eval_file}")

# Main evaluation loop
def evaluate_models(saved_models_folder, val_loader, custom_model_class, device):
    evaluation_metrics = []

    for model_filename in os.listdir(saved_models_folder):
        if model_filename.endswith(".pth"):
            print(f"Evaluating model: {model_filename}")
            
            # Load the saved model
            model_path = os.path.join(saved_models_folder, model_filename)
            model = load_model(model_path, custom_model_class, device)
            
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

model_decision = int(input("Choose desired models to evaluate (0 = classifier or 1 = autoencoder): "))
if model_decision == 0:
    # Load validation dataset (same as in the training script)
    annotations_file = pd.read_csv("TRAIN_DATA.csv")  # Or your validation set CSV
    data_dir = "USABLE"
    img_list = LoadAnnotated(annotations_file, data_dir)
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = StandardImageDataset(annotations_file, img_list, transform=transform)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)  # Adjust batch size as needed

    # Define the device (CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Folder containing the saved models
    saved_models_folder = "saved_models"

    # Start evaluating all models
    evaluate_models(saved_models_folder, val_loader, CustomModel, device)
