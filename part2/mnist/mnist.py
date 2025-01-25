import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Load CSV data (replace with actual file paths)
data = pd.read_csv("train.csv")

# Split the data into training and test sets
train_data, test_data = data.iloc[:-100], data.iloc[-100:]

# Assuming the CSV contains the features and the label in columns
# Features (X) and Labels (y) are assumed to be in columns 'feature1', 'feature2', ..., 'featureN' and 'label'
X_train = train_data.drop(columns=["label"]).values
y_train = train_data["label"].values
X_test = test_data.drop(columns=["label"]).values
y_test = test_data["label"].values

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Xtrain class: {X_train.__class__}, ytrain class: {y_train.__class__}")
print(f"Xtrain dtype: {X_train.dtype}, ytrain dtype: {y_train.dtype}")
print(f"Xtrain shape: {X_train.shape}, ytrain shape: {y_train.shape}")

# plot 3 random samples in terminal
import random
for _id in random.sample(range(X_train.shape[0]), 3):
    arr = X_train[_id]
    
    for i in range(28):
        for j in range(28):
            print('#' if (arr[i*28+j]>100) else '.', end='')
        print()
    print('\n' + y_train[_id].astype(str))

# cast to float
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.int64)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.int64)

# Standardize the features (mean=0, std=1)
X_train /=  255.
X_test /= 255.

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model, loss, and optimizer
input_size = X_train.shape[1]
hidden_size = 128
output_size = len(np.unique(y_train))  # number of unique classes

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, X_train, y_train, epochs=5, batch_size=64):
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Inference
def inference(model, X):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
    return predicted

# Train the model
train(model, X_train_tensor, y_train_tensor)

# Test the model
y_pred = inference(model, X_test_tensor)
accuracy = (y_pred == y_test_tensor).float().mean().item()
print(f"Test Accuracy: {accuracy * 100:.2f}%")

