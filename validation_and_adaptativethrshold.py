import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from train_utils import LoadAnnotated, StandardImageDataset
from torch.utils.data import DataLoader, Subset

# Define function to calculate patient-level predictions and ROC threshold
def evaluate_and_threshold(model_path, val_loader, device):
    # Load the trained model
    model = CustomModel(...)  # Define the model architecture
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize dictionaries for predictions
    patient_predictions = {}
    patient_labels = {}

    # Predict on validation set
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Positive class probabilities

            # Store predictions by patient ID
            for i, prob in enumerate(probs):
                pat_id = val_loader.dataset.img_labels.iloc[i]['Pat_ID']
                if pat_id not in patient_predictions:
                    patient_predictions[pat_id] = []
                    patient_labels[pat_id] = labels.cpu().numpy()[i]  # Assuming all patches have same patient label
                patient_predictions[pat_id].append(prob)

    # Aggregate to patient level
    mean_scores = np.array([np.mean(preds) for preds in patient_predictions.values()])
    ground_truth_labels = np.array(list(patient_labels.values()))

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth_labels, mean_scores)
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

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)  # Example criterion
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold}")

    return optimal_threshold

val_data = pd.read_csv("validation_data.csv")
img_list = LoadAnnotated(val_data, "data_directory")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
val_dataset = StandardImageDataset(val_data, img_list, transform=transform)

# Set up DataLoader
val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Specify the path to the saved model
model_path = "saved_models/your_model_name.pth"

# Evaluate model and calculate optimal threshold
optimal_threshold = evaluate_and_threshold(model_path, val_loader, device)
