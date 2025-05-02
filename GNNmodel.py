# Import necessary libraries
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random


# Set random seeds for reproducibility of results
seed = 42
random.seed(seed)  # Python random seed
np.random.seed(seed)  # NumPy random seed
torch.manual_seed(seed)  # PyTorch CPU random seed
torch.cuda.manual_seed(seed)  # PyTorch GPU random seed
torch.cuda.manual_seed_all(seed)  # Seed for all GPUs
torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior on GPUs
torch.backends.cudnn.benchmark = False  # Disables non-deterministic algorithms


# Define label mapping for seizure categories
labels_map = {'No_Seizure': 0, 'P': 1, 'PG': 2}


# Preprocessing pipeline: Resize, Augment, Normalize, and Convert to Tensor
pretransform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy image to PIL image
    transforms.RandomHorizontalFlip(),  # Randomly flip image horizontally for augmentation
    transforms.RandomRotation(20),  # Randomly rotate image by up to 20 degrees
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop to 224x224 with scaling
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])


# Function to load image frames from folders
def load_frames_from_folders(base_folder_path):
    frames = []  # To store image frames
    labels = []  # To store corresponding labels

    # Loop over each folder in the base directory
    for folder_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, folder_name)

        if not os.path.isdir(folder_path):
            continue  # Skip if not a directory

        label = labels_map.get(folder_name)
        if label is None:
            continue  # Skip if label not found

        # Loop over each patient's folder within the label folder
        for patient_folder in os.listdir(folder_path):
            patient_folder_path = os.path.join(folder_path, patient_folder)

            if not os.path.isdir(patient_folder_path):
                continue  # Skip if not a directory

            # Loop through each image file in the patient folder
            for filename in os.listdir(patient_folder_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):  # Image extensions
                    img = cv2.imread(os.path.join(patient_folder_path, filename))  # Read image
                    img = cv2.resize(img, (224, 224))  # Resize image to 224x224

                    # Apply preprocessing transformations (augmentation + normalization)
                    img = pretransform(img)

                    frames.append(img)  # Add image to frames list
                    labels.append(label)  # Add label to labels list

    return frames, np.array(labels)


# Load frames from the given directory
frames, labels = load_frames_from_folders('/content/drive/MyDrive/frames')

# Stack frames and convert labels to a tensor
frames = torch.stack(frames)
labels = torch.tensor(labels, dtype=torch.long)

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(frames, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert data to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for each split (train, validation, test)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Load a pre-trained ResNet18 model and modify the final layer to output 512 features
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 512)  # Modify final layer


# Graph Convolution Layer for Graph Neural Network (GNN)
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))  # Weight matrix
        nn.init.xavier_uniform_(self.weight)  # Initialize weights with Xavier uniform distribution

    def forward(self, adj_matrix, feature_matrix):
        # Perform graph convolution: multiply feature matrix by weight, then apply adjacency matrix
        output = torch.mm(feature_matrix, self.weight)
        output = torch.mm(adj_matrix, output)
        return output


# GNN combined with CNN (ResNet) for feature extraction and graph-based learning
class GNNWithCNN(nn.Module):
    def __init__(self, cnn_model, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GNNWithCNN, self).__init__()
        self.cnn = cnn_model  # Pretrained CNN model (ResNet)
        self.gc1 = GraphConvolution(input_dim, hidden_dim)  # First GNN layer
        self.gc2 = GraphConvolution(hidden_dim, output_dim)  # Second GNN layer
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(dropout_rate)  # Dropout for regularization

    def forward(self, adj_matrix, feature_matrix):
        batch_size = feature_matrix.size(0)
        feature_matrix = feature_matrix.view(batch_size, 3, 224, 224)  # Reshape to (batch_size, 3, 224, 224)

        # Extract features using the CNN (ResNet)
        features = self.cnn(feature_matrix)

        hidden = self.gc1(adj_matrix, features)  # Apply GNN to extracted features
        hidden = self.relu(hidden)  # Apply ReLU activation
        hidden = self.dropout(hidden)  # Apply dropout
        output = self.gc2(adj_matrix, hidden)  # Apply second GNN layer
        return output


# Initialize the GNN with CNN model
model = GNNWithCNN(cnn_model=resnet, input_dim=512, hidden_dim=128, output_dim=3, dropout_rate=0.5)

# Calculate class weights to handle class imbalance
labels_np = labels.numpy()
unique_labels = np.unique(labels_np)  # Find unique labels
class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels_np)  # Compute weights
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)  # CrossEntropyLoss with class weights
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

num_epochs = 10  # Set number of training epochs
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0  # Variable to accumulate loss

    # Training loop for each batch in the train_loader
    for batch in train_loader:
        features, labels = batch
        optimizer.zero_grad()  # Zero out gradients

        # Forward pass with graph attention
        adj_matrix = torch.eye(len(features))  # Identity matrix as adjacency matrix (simple graph structure)
        output = model(adj_matrix, features.view(len(features), -1))  # Flatten input features for GNN

        loss = criterion(output, labels)  # Compute loss

        loss.backward()  # Backward pass to compute gradients
        optimizer.step()  # Update model weights

        running_loss += loss.item()  # Accumulate loss for this epoch

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # No need to compute gradients during validation
        val_loss = 0.0
        for batch in val_loader:
            features, labels = batch
            adj_matrix = torch.eye(len(features))  # Identity matrix as adjacency matrix
            output = model(adj_matrix, features.view(len(features), -1))
            loss = criterion(output, labels)
            val_loss += loss.item()

        print(f"Validation Loss after epoch {epoch+1}: {val_loss/len(val_loader)}")

    # Update the learning rate based on validation loss
    scheduler.step(val_loss / len(val_loader))
