import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score
from pathlib import Path
import os

# Reconstruct light curve using a two-harmonic Fourier series
def reconstruct_light_curve_full_fourier(M, Per, Amp_r, R21, phi21, num_points):
    phase = np.linspace(0, 2, num_points)
    A1 = 0.5 * Amp_r
    A2 = A1 * R21
    a1 = A1 * np.cos(0)
    b1 = A1 * np.sin(0)
    a2 = A2 * np.cos(phi21)
    b2 = A2 * np.sin(phi21)
    return (M + a1 * np.cos(2 * np.pi * phase) + b1 * np.sin(2 * np.pi * phase) +
            a2 * np.cos(4 * np.pi * phase) + b2 * np.sin(4 * np.pi * phase))

# CNN-LSTM Model definition
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN_LSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv1d(256, 256, 5, padding=2)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.lstm1 = nn.LSTM(256, 64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)                # [B, 1, L]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        x = x.permute(0, 2, 1)              # [B, new_L, features]
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]      
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Setup
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_length = 359
num_classes = 5

fold_results = []
all_f1_scores = []
all_precisions = []
all_recalls = []
all_accuracy = []

# 5-fold cross-validation loop
for fold in range(1, 6):
    train_df = pd.read_csv(Path(os.path.join(data_path, f"inf_lstm_train_fold_{fold}.txt")), sep=' ')
    test_df = pd.read_csv(Path(os.path.join(data_path, f"inf_lstm_test_fold_{fold}.txt")), sep=' ')
    
    X_train_list, y_train = [], []
    for _, row in train_df.iterrows():
        X_train_list.append(reconstruct_light_curve_full_fourier(
            row['rmag'], row['Per'], row['Amp_r'], row['R21'], row['phi21'], num_points=input_length))
        y_train.append(row['Type'])
    X_train = np.stack(X_train_list, axis=0)
    
    X_test_list, y_test = [], []
    for _, row in test_df.iterrows():
        X_test_list.append(reconstruct_light_curve_full_fourier(
            row['rmag'], row['Per'], row['Amp_r'], row['R21'], row['phi21'], num_points=input_length))
        y_test.append(row['Type'])
    X_test = np.stack(X_test_list, axis=0)
    
    y_train = pd.Categorical(y_train).codes
    y_test = pd.Categorical(y_test).codes
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = CNN_LSTM_Model(input_length=input_length, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    num_epochs = 100

    # Training
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
        if (epoch + 1) % 25 == 0:
            print(f"Fold {fold}, Epoch {epoch+1}: Loss {running_loss/total:.4f}, Accuracy {correct/total:.4f}")
    
    # Evaluation
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
    test_accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    fold_results.append((test_accuracy, f1, precision, recall))
    all_accuracy.append(test_accuracy)
    all_f1_scores.append(f1)
    all_precisions.append(precision)
    all_recalls.append(recall)
    print(f"Fold {fold} - Accuracy: {test_accuracy*100:.2f}%, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

avg_accuracy = np.mean(all_accuracy)
avg_f1 = np.mean(all_f1_scores)
avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)

print("\n=== Cross-Validation Summary ===")
print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
print(f"Average F1: {avg_f1:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")




#Fold 1, Epoch 25: Loss 0.3507, Accuracy 0.8737
#Fold 1, Epoch 50: Loss 0.3038, Accuracy 0.8908
#Fold 1, Epoch 75: Loss 0.2932, Accuracy 0.8948
#Fold 1, Epoch 100: Loss 0.2864, Accuracy 0.8973
#Fold 1 - Accuracy: 90.07%, F1: 0.8975, Precision: 0.8991, Recall: 0.9007
#Fold 2, Epoch 25: Loss 0.3588, Accuracy 0.8690
#Fold 2, Epoch 50: Loss 0.3042, Accuracy 0.8927
#Fold 2, Epoch 75: Loss 0.2928, Accuracy 0.8951
#Fold 2, Epoch 100: Loss 0.2863, Accuracy 0.8980
#Fold 2 - Accuracy: 90.00%, F1: 0.8963, Precision: 0.8953, Recall: 0.9000
#Fold 3, Epoch 25: Loss 0.3488, Accuracy 0.8744
#Fold 3, Epoch 50: Loss 0.3042, Accuracy 0.8919
#Fold 3, Epoch 75: Loss 0.2921, Accuracy 0.8961
#Fold 3, Epoch 100: Loss 0.2865, Accuracy 0.8979
#Fold 3 - Accuracy: 90.04%, F1: 0.8973, Precision: 0.8976, Recall: 0.9004
#Fold 4, Epoch 25: Loss 0.3389, Accuracy 0.8780
#Fold 4, Epoch 50: Loss 0.3015, Accuracy 0.8925
#Fold 4, Epoch 75: Loss 0.2928, Accuracy 0.8952
#Fold 4, Epoch 100: Loss 0.2864, Accuracy 0.8982
#Fold 4 - Accuracy: 90.18%, F1: 0.8978, Precision: 0.8990, Recall: 0.9018
#Fold 5, Epoch 25: Loss 0.3225, Accuracy 0.8859
#Fold 5, Epoch 50: Loss 0.2996, Accuracy 0.8937
#Fold 5, Epoch 75: Loss 0.2903, Accuracy 0.8970
#Fold 5, Epoch 100: Loss 0.2858, Accuracy 0.8990
#Fold 5 - Accuracy: 89.89%, F1: 0.8948, Precision: 0.8984, Recall: 0.8989

#=== Cross-Validation Summary ===
#Average Accuracy: 90.04%
#Average F1: 0.8967
#Average Precision: 0.8979
#Average Recall: 0.9004


# Fold 1, Epoch 25: Loss 0.3290, Accuracy 0.8827
# Fold 1, Epoch 50: Loss 0.2803, Accuracy 0.8984
# Fold 1, Epoch 75: Loss 0.2716, Accuracy 0.9009
# Fold 1, Epoch 100: Loss 0.2662, Accuracy 0.9029
# Fold 1 - Accuracy: 90.30%, F1: 0.8988, Precision: 0.8996, Recall: 0.9030
# Fold 2, Epoch 25: Loss 0.2838, Accuracy 0.8984
# Fold 2, Epoch 50: Loss 0.2688, Accuracy 0.9029
# Fold 2, Epoch 75: Loss 0.2617, Accuracy 0.9034
# Fold 2, Epoch 100: Loss 0.2556, Accuracy 0.9058
# Fold 2 - Accuracy: 90.17%, F1: 0.8991, Precision: 0.8993, Recall: 0.9017
# Fold 3, Epoch 25: Loss 0.2862, Accuracy 0.8970
# Fold 3, Epoch 50: Loss 0.2729, Accuracy 0.9009
# Fold 3, Epoch 75: Loss 0.2658, Accuracy 0.9032
# Fold 3, Epoch 100: Loss 0.2593, Accuracy 0.9043
# Fold 3 - Accuracy: 89.55%, F1: 0.8920, Precision: 0.8936, Recall: 0.8955
# Fold 4, Epoch 25: Loss 0.2880, Accuracy 0.8952
# Fold 4, Epoch 50: Loss 0.2749, Accuracy 0.8998
# Fold 4, Epoch 75: Loss 0.2673, Accuracy 0.9021
# Fold 4, Epoch 100: Loss 0.2612, Accuracy 0.9042
# Fold 4 - Accuracy: 90.26%, F1: 0.8998, Precision: 0.8997, Recall: 0.9026
# Fold 5, Epoch 25: Loss 0.2869, Accuracy 0.8968
# Fold 5, Epoch 50: Loss 0.2745, Accuracy 0.9007
# Fold 5, Epoch 75: Loss 0.2657, Accuracy 0.9035
# Fold 5, Epoch 100: Loss 0.2604, Accuracy 0.9052
# Fold 5 - Accuracy: 90.00%, F1: 0.8963, Precision: 0.8998, Recall: 0.9000
#
# === Cross-Validation Summary ===
# Average Accuracy: 90.06%
# Average F1: 0.8972
# Average Precision: 0.8984%
# Average Recall: 0.9006%
