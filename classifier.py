import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

import torchvision
import os

# Load FashionMNIST from local files
# DATA_DIR is set to current directory where files are stored
DATA_DIR = "."
download_dataset = False  # No download needed; load from local

# Load training and test datasets from torchvision
train_FashionMnist = datasets.FashionMNIST(DATA_DIR, train=True, download=download_dataset)
test_FashionMnist = datasets.FashionMNIST(DATA_DIR, train=False, download=download_dataset)

# Display dataset info for debugging/verification
print("hello Assignment1")
print(train_FashionMnist)
print(test_FashionMnist)

# Show the shape of the data (number of images and image dimensions)
print("\nthe shape of the data and targets")
print(train_FashionMnist.data.shape)   # e.g., (60000, 28, 28)
print(train_FashionMnist.targets.shape)
print(test_FashionMnist.data.shape)    # e.g., (10000, 28, 28)
print(test_FashionMnist.targets.shape)

# Convert image tensors to float for normalization and further processing
X_train = train_FashionMnist.data.float()
y_train = train_FashionMnist.targets
X_test = test_FashionMnist.data.float()
y_test = test_FashionMnist.targets

# ======================
# Create Validation Set
# ======================
# Use 10,000 random samples from training data as validation set
test_size = X_test.shape[0]  # same size as test set
indices = np.random.choice(X_train.shape[0], test_size, replace=False)

X_valid = X_train[indices]
y_valid = y_train[indices]

# Remove validation samples from training data
X_train = np.delete(X_train, indices, axis=0)
y_train = np.delete(y_train, indices, axis=0)

# Print updated sizes after validation split
print("\nfinal data sizes")
print(X_train.shape)  # 50,000 samples
print(y_train.shape)
print(X_valid.shape)  # 10,000 samples
print(y_valid.shape)
print(X_test.shape)   # 10,000 samples (unchanged)
print(y_test.shape)

# ======================
# Preprocessing
# ======================
# Flatten each 28x28 image into a 784-element vector for ANN input
X_train = X_train.reshape(-1, 28*28)
X_valid = X_valid.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

# Normalize pixel values to [0, 1] range
X_train /= 255.0
X_valid /= 255.0
X_test /= 255.0

# ======================
# Create PyTorch Datasets and DataLoaders
# ======================
# Wrap data into TensorDataset objects
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

# Create DataLoaders for batching and shuffling
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define a fully connected ANN with 3 hidden layers and dropout regularization
class ANN(nn.Module):
    def __init__(self, input_size, h1, h2, h3, num_classes, dropout_rate=0.5):
        super(ANN, self).__init__()
        
        # First hidden layer (input_size → h1)
        self.fc1 = nn.Linear(input_size, h1)
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout after layer 1
        
        # Second hidden layer (h1 → h2)
        self.fc2 = nn.Linear(h1, h2)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout after layer 2
        
        # Third hidden layer (h2 → h3)
        self.fc3 = nn.Linear(h2, h3)
        self.dropout3 = nn.Dropout(dropout_rate)  # Dropout after layer 3
        
        # Output layer (h3 → num_classes)
        self.fc4 = nn.Linear(h3, num_classes)

    def forward(self, x):
        # Forward pass through hidden layers with ReLU activation + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        # Final output (logits for 10 classes)
        x = self.fc4(x)
        return x


# ==========================
# Training Configuration
# ==========================

num_classes = 10           # 10 FashionMNIST classes
batch_size = 64            # Size of data batches during training
h1 = 512                   # Hidden layer 1 nodes
h2 = 256                   # Hidden layer 2 nodes
h3 = 128                   # Hidden layer 3 nodes
input_size = 28 * 28       # Flattened 28x28 image = 784 features

dropout_rate = 0.2         # Regularization via dropout
learning_rate = 0.001      # Learning rate for optimizer

# Initialize model, optimizer, and loss settings
model = ANN(input_size, h1, h2, h3, num_classes, dropout_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 50            # Maximum number of training epochs

# ==========================
# Early Stopping Setup
# ==========================
patience = 7               # Stop if no improvement for 'patience' epochs
best_val_loss = float('inf')
counter = 0                # Tracks epochs without improvement

# ==========================
# Training Loop
# ==========================
for epoch in range(num_epochs):
    model.train()  # Enable dropout and batch norm if used

    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)                      # Forward pass
        labels = labels.long()                       # Ensure labels are correct type
        loss = F.cross_entropy(outputs, labels)      # Compute multi-class loss

        optimizer.zero_grad()                        # Clear gradients
        loss.backward()                              # Backpropagation
        optimizer.step()                             # Update weights

        # Print progress every 100 steps
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

    # ======================
    # Validation Phase
    # ======================
    model.eval()  # Disable dropout
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():  # No gradient calculation for validation
        for images, labels in val_loader:
            outputs = model(images)
            labels = labels.long()
            loss = F.cross_entropy(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Take class with highest score
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)  # Average validation loss
    val_accuracy = 100 * val_correct / val_total
    print(f'Validation Loss after Epoch {epoch+1}: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

    # ======================
    # Early Stopping Logic
    # ======================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0                       # Reset counter if model improved
        best_model = model.state_dict()  # Save model checkpoint
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            model.load_state_dict(best_model)  # Restore best model
            break  # Exit training loop early


# ==========================
# Final Test Accuracy Evaluation
# ==========================
model.eval()  # Set model to evaluation mode (disables dropout)
with torch.no_grad():  # No gradient calculation needed for evaluation
    correct = 0
    total = 0

    # Loop through test data
    for images, labels in test_loader:
        outputs = model(images)  # Forward pass on test data
        _, predicted = torch.max(outputs.data, 1)  # Take the class with highest logit score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # Count correct predictions

    # Print test accuracy as a percentage
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# ==========================
# Optional: Preview number Random Predictions
# ==========================
number = 0 # number of Random Predictions
model.eval()
with torch.no_grad():
    for i in range(number):
        # Randomly select a test image
        idx = random.randint(0, X_test.shape[0])
        image = X_test[idx]  # Retrieve image
        label = y_test[idx]  # True label

        # Flatten image and pass through model
        output = model(torch.tensor(image).view(1, -1))
        _, predicted = torch.max(output, 1)

        # Uncomment for visual debugging
        if label == predicted:
            # print("True label: %d, Predicted label: %d correct" % (label, predicted))
            pass
        else:
            # print("True label: %d, Predicted label: %d wrong" % (label, predicted))
            pass

print("Done!\n")

# ==========================
# Class Inference Tool (CLI)
# ==========================

# Human-readable labels for FashionMNIST classes
class_names = [
    "T-shirt/top", 
    "Trouser", 
    "Pullover", 
    "Dress", 
    "Coat",
    "Sandal", 
    "Shirt", 
    "Sneaker", 
    "Bag", 
    "Ankle boot"
]

# Interactive CLI loop
while True:
    jpg_path = input("Please enter a filepath:\n> ")

    if jpg_path.lower() == "exit":
        print("Exiting...")
        break  # Exit loop if user types 'exit'

    if not os.path.isfile(jpg_path):
        print("File does not exist. Try again.")
        continue  # Ask for a new file path if invalid

    try:
        # Load image in grayscale mode
        img = torchvision.io.read_image(jpg_path, mode=torchvision.io.ImageReadMode.GRAY)

        # Resize image to 28x28 to match training format
        img = torchvision.transforms.functional.resize(img, [28, 28])

        # Normalize pixel values to [0, 1]
        img = img.float() / 255.0

        # Flatten to a 784-element vector
        img = img.view(1, -1)

        # Model inference
        model.eval()
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)  # Get class index with highest logit
            print("Classifier:", class_names[predicted.item()])  # Map to readable class name

    except Exception as e:
        print(f"Error loading or processing image: {e}")
