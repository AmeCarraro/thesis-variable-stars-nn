import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from pathlib import Path
import time
import os

# Function to reconstruct light curves using Fourier series (2 harmonics)
def reconstruct_light_curve_full_fourier(M, Per, Amp_r, R21, phi21, num_points):
    phase = np.linspace(0, 2, num_points)
    A1 = 0.5 * Amp_r
    A2 = A1 * R21
    phi1 = 0
    phi2 = phi21
    a1 = A1 * np.cos(phi1)
    b1 = A1 * np.sin(phi1)
    a2 = A2 * np.cos(phi2)
    b2 = A2 * np.sin(phi2)
    # Sum of Fourier components
    light_curve = (M +
                   a1 * np.cos(2 * np.pi * phase) +
                   b1 * np.sin(2 * np.pi * phase) +
                   a2 * np.cos(4 * np.pi * phase) +
                   b2 * np.sin(4 * np.pi * phase))
    return light_curve

# CNN-LSTM hybrid model definition
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN_LSTM_Model, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        # Fully connected layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set data path and device (GPU if available)
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tracking lists
fold_results = []
all_f1_scores = []
all_precisions = []
all_recalls = []
all_accuracy = []

# Light curve length and number of classes
input_length = 359
num_classes = 5

# 5-Fold Cross-Validation
for fold in range(1, 6):
    # Load train/test data
    train_df = pd.read_csv(Path(os.path.join(data_path, f"5_lstm_train_fold_{fold}.txt")), sep=' ')
    test_df = pd.read_csv(Path(os.path.join(data_path, f"5_lstm_test_fold_{fold}.txt")), sep=' ')
    
    # Reconstruct light curves from features (train set)
    X_train_list = []
    y_train = []
    for idx, row in train_df.iterrows():
        lc = reconstruct_light_curve_full_fourier(
            row['rmag'], row['Per'], row['Amp_r'], row['R21'], row['phi21'], input_length
        )
        X_train_list.append(lc)
        y_train.append(row['Type'])
    X_train = np.stack(X_train_list, axis=0)

    # Reconstruct light curves from features (test set)
    X_test_list = []
    y_test = []
    for idx, row in test_df.iterrows():
        lc = reconstruct_light_curve_full_fourier(
            row['rmag'], row['Per'], row['Amp_r'], row['R21'], row['phi21'], input_length
        )
        X_test_list.append(lc)
        y_test.append(row['Type'])
    X_test = np.stack(X_test_list, axis=0)

    # Encode target classes as integers
    y_train = pd.Categorical(y_train).codes
    y_test = pd.Categorical(y_test).codes

    # Convert to tensors and move to device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function and optimizer
    model = CNN_LSTM_Model(input_length=input_length, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    num_epochs = 100
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        # Print every 25 epochs
        if (epoch + 1) % 25 == 0:
            print(f"Fold {fold}, Epoch {epoch+1}: Loss {running_loss/total:.4f}, Accuracy {correct/total:.4f}")

    # Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    test_accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    # Print results for the fold
    print(f"Fold {fold} - Confusion Matrix:")
    print(cm)
    fold_results.append((test_accuracy, f1, precision, recall))
    all_accuracy.append(test_accuracy)
    all_f1_scores.append(f1)
    all_precisions.append(precision)
    all_recalls.append(recall)
    print(f"Fold {fold} - Accuracy: {test_accuracy*100:.2f}%, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Print overall cross-validation results
avg_accuracy = np.mean(all_accuracy)
avg_f1 = np.mean(all_f1_scores)
avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)

print("\n=== Cross-Validation Summary ===")
print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
print(f"Average F1: {avg_f1:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")





# Fold 1, Epoch 25: Loss 1.5028, Accuracy 0.3088
# Fold 1, Epoch 50: Loss 0.7739, Accuracy 0.6913
# Fold 1, Epoch 75: Loss 0.6336, Accuracy 0.7450
# Fold 1, Epoch 100: Loss 0.5788, Accuracy 0.7672
# Fold 1 - Confusion Matrix:
# [[143  24   0  29  20]
#  [  5 152   0  27   6]
#  [  0   0 187   5   0]
#  [ 36  30  12 131   1]
#  [ 23  17   0   2 150]]
# Fold 1 - Accuracy: 76.30%, F1: 0.7618, Precision: 0.7637, Recall: 0.7630
# Fold 2, Epoch 25: Loss 1.4965, Accuracy 0.3118
# Fold 2, Epoch 50: Loss 0.8190, Accuracy 0.6590
# Fold 2, Epoch 75: Loss 0.6662, Accuracy 0.7230
# Fold 2, Epoch 100: Loss 0.6340, Accuracy 0.7428
# Fold 2 - Confusion Matrix:
# [[124  16   0  29  24]
#  [ 16 176   0  19  17]
#  [  0   1 200   8   1]
#  [ 40  22   9 103   4]
#  [ 21   8   0   2 160]]
# Fold 2 - Accuracy: 76.30%, F1: 0.7620, Precision: 0.7622, Recall: 0.7630
# Fold 3, Epoch 25: Loss 1.4934, Accuracy 0.3078
# Fold 3, Epoch 50: Loss 1.4732, Accuracy 0.3252
# Fold 3, Epoch 75: Loss 0.6378, Accuracy 0.7395
# Fold 3, Epoch 100: Loss 0.6155, Accuracy 0.7532
# Fold 3 - Confusion Matrix:
# [[ 95  23   1  32  35]
#  [ 13 182   0   5   3]
#  [  0   0 198   4   0]
#  [ 27  50  17 100   8]
#  [ 18  17   0   2 170]]
# Fold 3 - Accuracy: 74.50%, F1: 0.7347, Precision: 0.7407, Recall: 0.7450
# Fold 4, Epoch 25: Loss 1.5169, Accuracy 0.2945
# Fold 4, Epoch 50: Loss 1.5039, Accuracy 0.3103
# Fold 4, Epoch 75: Loss 0.6981, Accuracy 0.7202
# Fold 4, Epoch 100: Loss 0.6175, Accuracy 0.7528
# Fold 4 - Confusion Matrix:
# [[109  26   0  66  15]
#  [ 11 151   0  13  13]
#  [  0   1 175   7   0]
#  [ 40  50  23  91   2]
#  [ 44   9   0   1 153]]
# Fold 4 - Accuracy: 67.90%, F1: 0.6739, Precision: 0.6753, Recall: 0.6790
# Fold 5, Epoch 25: Loss 1.4986, Accuracy 0.2963
# Fold 5, Epoch 50: Loss 0.7367, Accuracy 0.7040
# Fold 5, Epoch 75: Loss 0.6547, Accuracy 0.7415
# Fold 5, Epoch 100: Loss 0.6111, Accuracy 0.7530
# Fold 5 - Confusion Matrix:
# [[117  21   0  27  24]
#  [ 10 164   0  12   5]
#  [  0   3 208   2   0]
#  [ 46  34   9 108   7]
#  [ 20  14   0   2 167]]
# Fold 5 - Accuracy: 76.40%, F1: 0.7597, Precision: 0.7644, Recall: 0.7640
# 
# === Cross-Validation Summary ===
# Average Accuracy: 74.28%
# Average F1: 0.7384
# Average Precision: 0.7412
# Average Recall: 0.7428

# Fold 1, Epoch 25: Loss 0.8322, Accuracy 0.6613
# Fold 1, Epoch 50: Loss 0.5878, Accuracy 0.7680
# Fold 1, Epoch 75: Loss 0.5162, Accuracy 0.7995
# Fold 1, Epoch 100: Loss 0.4793, Accuracy 0.8107
# Fold 1 - Confusion Matrix:
# [[112  33   0  41  30]
#  [  2 179   0   3   6]
#  [  0   0 189   3   0]
#  [ 17  49  10 133   1]
#  [  3  28   0   2 159]]
# Fold 1 - Accuracy: 77.20%, F1: 0.7657, Precision: 0.7898, Recall: 0.7720
# Fold 2, Epoch 25: Loss 1.4876, Accuracy 0.3157
# Fold 2, Epoch 50: Loss 1.1423, Accuracy 0.5182
# Fold 2, Epoch 75: Loss 0.6160, Accuracy 0.7618
# Fold 2, Epoch 100: Loss 0.5442, Accuracy 0.7890
# Fold 2 - Confusion Matrix:
# [[135  18   0  23  17]
#  [  9 197   0   3  19]
#  [  0   4 205   1   0]
#  [ 41  31  10  94   2]
#  [ 18  11   0   2 160]]
# Fold 2 - Accuracy: 79.10%, F1: 0.7862, Precision: 0.7911, Recall: 0.7910
# Fold 3, Epoch 25: Loss 1.4934, Accuracy 0.3083
# Fold 3, Epoch 50: Loss 0.6675, Accuracy 0.7312
# Fold 3, Epoch 75: Loss 0.5038, Accuracy 0.8077
# Fold 3, Epoch 100: Loss 0.4620, Accuracy 0.8133
# Fold 3 - Confusion Matrix:
# [[129   3   0  33  21]
#  [ 16 174   0  10   3]
#  [  0   0 192  10   0]
#  [ 31  14   5 148   4]
#  [ 19  18   0   0 170]]
# Fold 3 - Accuracy: 81.30%, F1: 0.8140, Precision: 0.8154, Recall: 0.8133
# Fold 4, Epoch 25: Loss 1.2775, Accuracy 0.4497
# Fold 4, Epoch 50: Loss 0.6633, Accuracy 0.7320
# Fold 4, Epoch 75: Loss 0.5315, Accuracy 0.7930
# Fold 4, Epoch 100: Loss 0.4933, Accuracy 0.8055
# Fold 4 - Confusion Matrix:
# [[147   6   0  42  21]
#  [ 11 128   0  26  23]
#  [  0   0 178   5   0]
#  [ 29  10   9 154   4]
#  [ 20   7   0   2 178]]
# Fold 4 - Accuracy: 78.50%, F1: 0.7842, Precision: 0.7885, Recall: 0.7850
# Fold 5, Epoch 25: Loss 1.2247, Accuracy 0.4795
# Fold 5, Epoch 50: Loss 0.6398, Accuracy 0.7442
# Fold 5, Epoch 75: Loss 0.5294, Accuracy 0.7860
# Fold 5, Epoch 100: Loss 0.4836, Accuracy 0.8105
# Fold 5 - Confusion Matrix:
# [[154   3   0  20  12]
#  [ 26 136   0   8  21]
#  [  0   1 207   5   0]
#  [ 55  10   9 126   4]
#  [ 46   7   0   1 149]]
# Fold 5 - Accuracy: 77.20%, F1: 0.7754, Precision: 0.7964, Recall: 0.7720
#
# === Cross-Validation Summary ===
# Average Accuracy: 78.66%
# Average F1: 0.7851
# Average Precision: 0.7962
# Average Recall: 0.7866
