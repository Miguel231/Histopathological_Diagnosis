import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from PIL import Image
import zipfile
import io
import random

# Configurations for the model
config = {
    'model': 'densenet201',     # Options: 'densenet201', 'resnet50', 'mobilenet_v2'
    'activation': 'relu',       # Options: 'relu', 'tanh', 'leaky_relu'
    'num_units': [1024, 512],   # Fully connected layer sizes
    'loss_fn': 'bce',           # Options: 'bce', 'bce_logits', 'focal'
    'optimizer': 'adam',        # Options: 'adam', 'sgd', 'adamw', 'rmsprop'
    'learning_rate': 0.001,
    'unfreeze_layers': False    # Set True to unfreeze specific model layers
}

# Step 1: Function to Load and Transform Images from ZIP (IGUAL PER TOTS ELS MODELS)
def load_images_from_zip(zip_path, transform, label_fn):
    images = []
    labels = []
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for file_name in zip_file.namelist():
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                with zip_file.open(file_name) as file:
                    image = Image.open(io.BytesIO(file.read())).convert('RGB')
                    images.append(transform(image))
                    labels.append(label_fn(file_name))
    return images, labels

# Step 2: Transformation Function (IGUAL PER TOTS ELS MODELS)
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Step 3: Labeling Function based on File Naming Convention
def label_fn(file_name):
    return 1 if 'h_pylori' in file_name else 0  # Adjust logic based on your file naming

# Step 4: Model Setup
def build_model(config):
    # Load the appropriate model based on config
    if config['model'] == 'densenet201':
        model = models.densenet201(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = nn.Identity()  # Remove the final classifier for custom layers
    
    elif config['model'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Identity()  # Remove the final fully connected layer for custom layers

    elif config['model'] == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Identity()  # Remove the final classifier for custom layers

    # Freeze layers based on configuration
    if not config['unfreeze_layers']:
        for param in model.parameters():
            param.requires_grad = False
    
    # Custom classifier
    layers = []
    activation_fn = nn.ReLU() if config['activation'] == 'relu' else nn.Tanh() if config['activation'] == 'tanh' else nn.LeakyReLU(0.01)
    
    for units in config['num_units']:
        layers.append(nn.Linear(num_features, units))
        layers.append(activation_fn)
        layers.append(nn.Dropout(0.5))
        num_features = units

    layers.append(nn.Linear(num_features, 1))
    if config['loss_fn'] == 'bce_logits':
        layers.append(nn.Identity())
    else:
        layers.append(nn.Sigmoid())

    if config['model'] == 'resnet50':
        model.fc = nn.Sequential(*layers)
    else:
        model.classifier = nn.Sequential(*layers)
    
    return model

# Step 5: Loss Function based on Config
def get_loss_function(config):
    if config['loss_fn'] == 'bce':
        return nn.BCELoss()
    elif config['loss_fn'] == 'bce_logits':
        return nn.BCEWithLogitsLoss()

# Step 6: Optimizer Setup
def get_optimizer(config, model):
    if config['optimizer'] == 'adam':
        return torch.optim.Adam(model.classifier.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        return torch.optim.SGD(model.classifier.parameters(), lr=config['learning_rate'], momentum=0.9)
    elif config['optimizer'] == 'adamw':
        return torch.optim.AdamW(model.classifier.parameters(), lr=config['learning_rate'], weight_decay=1e-4)

# Step 7: Training Function
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Step 8: Prepare Dataset
zip_path = 'path_to_your_zip_file.zip'  # Replace with actual path to your ZIP file
transform = get_transform()
images, labels = load_images_from_zip(zip_path, transform, label_fn)

# Step 9: DataLoader Setup
batch_size = 32
dataset = list(zip(images, labels))  # Combine images and labels into tuples
random.shuffle(dataset)               # Shuffle the dataset
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 10: Model Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = build_model(config)
criterion = get_loss_function(config)
optimizer = get_optimizer(config, model)

train_model(model, dataloader, criterion, optimizer, device)
