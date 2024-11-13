import os
import torch
import pandas as pd
import numpy as np
import PIL
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from classifier import CustomModel
import seaborn as sns
from PIL import Image
import warnings
from model_config import all_configurations
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
            try:
                os.remove(file_path)
                print(f"Deleted corrupted file: {file_path}")
            except Exception as delete_error:
                print(f"Error deleting file {file_path}: {delete_error}")    
    return IMS

def LoadAnnotated_1(df, data_dir):
    images = []
    labels = []
    
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        pat_id = row['CODI']  # Extract Pat_ID (patient identifier)
        print(f"Processing patient ID: {pat_id}")
        label = row['DENSITAT']  # Extract DENSITY (patient diagnosis/label)
        
        # Loop through each file in the directory
        for filename in os.listdir(data_dir):
            # Check if the filename starts with the Pat_ID (assuming the pattern 'Pat_ID_*')
            if filename.startswith(f"{pat_id}_"):
                file_path = os.path.join(data_dir, filename)
                
                # Check if the file exists
                if os.path.exists(file_path):
                    try:
                        # Open the image and convert to RGB
                        with Image.open(file_path) as img:
                            img = img.convert("RGB")
                            images.append(img)
                            labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {file_path}: {e}")
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
    #model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

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

def evaluate_model_with_autoencoder(model, val_loader, device):
    model.eval()  # Ensure model is in evaluation mode
    all_reconstruction_errors = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            reconstructed = model(images)  # Autoencoder's output (reconstructed images)

            # Compute reconstruction error (e.g., MSE)
            reconstruction_error = torch.mean((reconstructed - images) ** 2, dim=(1, 2, 3))  # MSE per image
            all_reconstruction_errors.extend(reconstruction_error.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    all_reconstruction_errors = np.array(all_reconstruction_errors)
    all_labels = np.array(all_labels)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_reconstruction_errors)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()

    # Find the optimal threshold (using the ROC curve)
    optimal_idx = np.argmax(tpr - fpr)  # Maximizing the difference between TPR and FPR
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold for Autoencoder: {optimal_threshold}")

    # Use optimal threshold to classify (thresholding reconstruction error)
    preds = (all_reconstruction_errors > optimal_threshold).astype(int)

    # Traditional metrics
    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds)
    recall = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    conf_matrix = confusion_matrix(all_labels, preds)

    return accuracy, precision, recall, f1, conf_matrix, roc_auc, optimal_threshold


def evaluate_model_with_classifier(model, val_loader, device):
    model.eval()  # Ensure model is in evaluation mode
    all_labels = []
    all_probs = []  # Probabilities of the positive class
    patient_predictions = {}
    patient_labels = {}

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Positive class probabilities

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

            # Store predictions by patient ID
            for i, prob in enumerate(probs):
                # Access img_labels through the original dataset
                pat_id = val_loader.dataset.dataset.img_labels.iloc[i]['Pat_ID']

                if pat_id not in patient_predictions:
                    patient_predictions[pat_id] = []
                    patient_labels[pat_id] = labels.cpu().numpy()[i]
                patient_predictions[pat_id].append(prob)

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()

    # Find the optimal threshold (using the ROC curve)
    optimal_idx = np.argmax(tpr - fpr)  # Maximizing the difference between TPR and FPR
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold for Classifier: {optimal_threshold}")

    # Use optimal threshold to classify (thresholding probabilities)
    preds = (all_probs > optimal_threshold).astype(int)

    # Traditional metrics
    accuracy = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds)
    recall = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    conf_matrix = confusion_matrix(all_labels, preds)

    return accuracy, precision, recall, f1, conf_matrix, roc_auc, optimal_threshold


def Patient_Diagnosis(saved_models_folder, save_folder, val_loader, device, mode):
    evaluation_metrics = []
    i = 0
    for model_filename in os.listdir(saved_models_folder):
        if model_filename.endswith(".pth"):
            print(f"Evaluating model: {model_filename}")
            
            # Load the saved model
            model_path = os.path.join(saved_models_folder, model_filename)
            configuration = all_configurations
            for config in configuration:
                model = load_model(model_path, device,config)

                # Evaluate model based on its type (autoencoder vs classifier)
                if mode == 1:  # Autoencoder
                    print("AUTOENCODER")
                    accuracy, precision, recall, f1, conf_matrix, roc_auc, optimal_threshold = evaluate_model_with_autoencoder(model, val_loader, device)
                elif mode == 0:  # Classifier
                    print("CLASSIFIER")
                    accuracy, precision, recall, f1, conf_matrix, roc_auc, optimal_threshold = evaluate_model_with_classifier(model, val_loader, device)
                    print("ACCURACY",accuracy)
                # Save evaluation metrics
                evaluation_results = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "confusion_matrix": [conf_matrix.tolist()]  # Convert to list for saving
                }
                
                if mode == "0":  # For classifier, also add ROC AUC and optimal threshold
                    evaluation_results["roc_auc"] = roc_auc
                    evaluation_results["optimal_threshold"] = optimal_threshold
                elif mode == "1":  # For autoencoder, also add ROC AUC and optimal threshold
                    evaluation_results["roc_auc"] = roc_auc
                    evaluation_results["optimal_threshold"] = optimal_threshold

                evaluation_metrics.append(evaluation_results)
                
                # Plot and save confusion matrix
                cm_save_path = os.path.join(save_folder, f"{model_filename}_confusion_matrix.png")
                plot_confusion_matrix(conf_matrix, cm_save_path)

    # Save all evaluation results to CSV
    evaluation_df = pd.DataFrame(evaluation_metrics)
    evaluation_df_file = os.path.join(save_folder, "all_evaluation_results.csv")
    evaluation_df.to_csv(evaluation_df_file, index=False)
    print(f"Saved all evaluation results to {evaluation_df_file}")


mode = int(input("PATIENT DIAGNOSIS (0 = classifier and 1 = autoencoder)?: "))

if mode == 0:
    annotated_csv = pd.read_csv(r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis-5\TRAIN_DATA_cropped.csv")
    # Load all images from `USABLE_annotated` using all_annotations
    data_dir = r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis-5\USABLE_cropped"
    img_list = LoadAnnotated_1(annotated_csv, data_dir)  # Assuming this function loads all images for the patches specified
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = StandardImageDataset(annotated_csv, img_list, transform=transform)
    # Set up folder to save models
    save_folder = "save_evaluations"
    os.makedirs(save_folder, exist_ok=True)
    # You can now filter the dataset again based on these indices if needed
    val_subset_indices = torch.load(r"C:\Users\larar\OneDrive\Documentos\Escritorio\Histopathological_Diagnosis-5\validation_and_training_data\val_subset_info_fold1.pth")
    val_subset = Subset(dataset, val_subset_indices['val_indices'])
    # Create the validation DataLoader using the same batch size and shuffle settings
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    # Define the device (CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved_models = "best_saved_models"
    # Start evaluating all models
    Patient_Diagnosis(saved_models,save_folder, val_loader, device, mode)
