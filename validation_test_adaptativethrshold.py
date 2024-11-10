# Assuming you have a test_loader for test dataset and a mean_threshold from cross-validation
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize a dictionary to store the predictions for each patient
patient_predictions = {}

# Loop over the test data (patches)
with torch.no_grad():
    custom_model.eval()  # Set the model to evaluation mode (model selection based on best performance in validation)
    for img, label in test_loader:
        img = img.to(device)
        img_flat = img.view(img.size(0), -1)  # Flatten the image (if needed for your model)
        
        # Get the model output (reconstructed image)
        output = custom_model(img_flat)
        
        # Calculate reconstruction errors on the test patches
        patch_errors = calculate_reconstruction_errors(model, patch_loader)  # Use the model and data loader for the test set

        # Classify patches based on the adaptive threshold
        predictions = []
        for error in patch_errors:
            is_anomalous = error > adaptive_threshold  # Compare error with the adaptive threshold
            predictions.append(is_anomalous)

        # Convert predictions to 0 (normal) or 1 (anomalous)
        patch_predictions = [1 if pred else 0 for pred in predictions]

        # Store patch predictions for each patient
        patient_predictions = {}  # {patient_id: list of patch-level predictions}

        for i, (img, label) in enumerate(patch_loader):  # Assuming you're iterating through the patches
            patient_id = img[0]  # Assuming patient_id is a part of the patch data (adjust based on your dataset)
            
            if patient_id not in patient_predictions:
                patient_predictions[patient_id] = []

            # Append patch prediction to the patient_id's list
            patient_predictions[patient_id].append(patch_predictions[i])


# Initialize a dictionary to store patient-level diagnosis
patient_diagnosis = {}

# Loop through the predictions
for patient_id, predictions in patient_predictions.items():
    # Count anomalies based on the adaptive threshold
    anomaly_count = sum(predictions)
    total_patches = len(predictions)
    
    # Calculate the anomaly percentage based on adaptive threshold
    anomaly_percentage = (anomaly_count / total_patches) * 100
    
    # Classify patient based on the anomaly percentage and adaptive threshold
    diagnosis = 1 if anomaly_percentage > mean_threshold * 100 else 0
    
    patient_diagnosis[patient_id] = {
        "diagnosis": diagnosis,
        "anomaly_percentage": anomaly_percentage
    }

# Print patient-level diagnosis
for patient_id, diagnosis_info in patient_diagnosis.items():
    print(f"Patient {patient_id}: {diagnosis_info['diagnosis']} (Anomaly percentage: {diagnosis_info['anomaly_percentage']:.2f}%)")
