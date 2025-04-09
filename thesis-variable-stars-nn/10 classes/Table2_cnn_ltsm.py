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

# Function to reconstruct a light curve using a full Fourier series with two harmonics
def reconstruct_light_curve_full_fourier(M, Per, Amp_r, R21, phi21, num_points):
    # Create equally spaced phases over one period (scaled to 0-2 for cosine/sine frequencies)
    phase = np.linspace(0, 2, num_points)
    # First harmonic amplitude is half of the provided amplitude
    A1 = 0.5 * Amp_r
    # Second harmonic amplitude is scaled by R21
    A2 = A1 * R21
    # Set phase offsets: first harmonic phase is 0, second follows phi21
    phi1 = 0
    phi2 = phi21
    # Compute Fourier coefficients for both harmonics
    a1 = A1 * np.cos(phi1)
    b1 = A1 * np.sin(phi1)
    a2 = A2 * np.cos(phi2)
    b2 = A2 * np.sin(phi2)
    # Reconstruct light curve as a sum of constant magnitude (M) and two harmonics
    light_curve = (M +
                   a1 * np.cos(2 * np.pi * phase) +
                   b1 * np.sin(2 * np.pi * phase) +
                   a2 * np.cos(4 * np.pi * phase) +
                   b2 * np.sin(4 * np.pi * phase))
    return light_curve

# Define the CNN-LSTM model class
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_length, num_classes):
        super(CNN_LSTM_Model, self).__init__()
        # Two convolutional layers with ReLU and padding, followed by max pooling
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Two additional convolutional layers with larger kernel size and more filters
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # Two LSTM layers to model temporal dependencies in the extracted features
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        # Fully connected layers for classification
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (batch_size, input_length); add channel dimension for Conv1D
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)
        # Permute to shape (batch, sequence_length, features) for LSTM layers
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # Use the output from the last time step
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Final output logits for each class
        return x

# Define data path, device, and hyperparameters
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fold_results = []
all_f1_scores = []
all_precisions = []
all_recalls = []
all_accuracy = []
input_length = 359  # Number of points in the reconstructed light curve
num_classes = 10    # Number of output classes

# Loop over the 5 cross-validation folds
for fold in range(1, 6):
    # Load the training and testing data for the current fold
    train_df = pd.read_csv(Path(os.path.join(data_path, f"lstm_train_fold_{fold}.txt")), sep=' ')
    test_df = pd.read_csv(Path(os.path.join(data_path, f"lstm_test_fold_{fold}.txt")), sep=' ')
    
    # Reconstruct light curves for training set
    X_train_list = []
    y_train = []
    for idx, row in train_df.iterrows():
        lc = reconstruct_light_curve_full_fourier(
            row['rmag'],  # Base magnitude
            row['Per'],   # Period
            row['Amp_r'], # Amplitude in r-band
            row['R21'],   # Fourier parameter R21
            row['phi21'], # Fourier phase parameter phi21
            num_points=input_length
        )
        X_train_list.append(lc)
        y_train.append(row['Type'])
    # Stack training light curves into a NumPy array
    X_train = np.stack(X_train_list, axis=0)
    
    # Reconstruct light curves for test set
    X_test_list = []
    y_test = []
    for idx, row in test_df.iterrows():
        lc = reconstruct_light_curve_full_fourier(
            row['rmag'],
            row['Per'],
            row['Amp_r'],
            row['R21'],
            row['phi21'],
            num_points=input_length
        )
        X_test_list.append(lc)
        y_test.append(row['Type'])
    X_test = np.stack(X_test_list, axis=0)
    
    # Convert categorical labels to numeric codes
    y_train = pd.Categorical(y_train).codes
    y_test = pd.Categorical(y_test).codes
    
    # Convert numpy arrays to torch tensors and move them to the chosen device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Create TensorDatasets and DataLoaders for training and testing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    model = CNN_LSTM_Model(input_length=input_length, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    
    num_epochs = 100
    # Training loop over epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # Iterate over mini-batches
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Reset gradients for the current batch
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item() * inputs.size(0)
            # Compute batch accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        # Print status every 25 epochs
        if (epoch + 1) % 25 == 0:
            print(f"Fold {fold}, Epoch {epoch+1}: Loss {running_loss/total:.4f}, Accuracy {correct/total:.4f}")
    
    # Evaluation on test data
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
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Fold {fold} - Confusion Matrix:")
    print(cm)
    fold_results.append((test_accuracy, f1, precision, recall))
    all_accuracy.append(test_accuracy)
    all_f1_scores.append(f1)
    all_precisions.append(precision)
    all_recalls.append(recall)
    print(f"Fold {fold} - Accuracy: {test_accuracy*100:.2f}%, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Calculate and print average metrics over all folds
avg_accuracy = np.mean(all_accuracy)
avg_f1 = np.mean(all_f1_scores)
avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)

print("\n=== Cross-Validation Summary ===")
print(f"Average Accuracy: {avg_accuracy*100:.2f}%")
print(f"Average F1: {avg_f1:.4f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")

# (The commented-out text at the end shows example output from different experiments.)




# Fold 1, Epoch 25: Loss 2.0390, Accuracy 0.2410
# Fold 1, Epoch 50: Loss 1.2816, Accuracy 0.5119
# Fold 1, Epoch 75: Loss 1.0033, Accuracy 0.6124
# Fold 1, Epoch 100: Loss 0.8818, Accuracy 0.6564
# Fold 1 - Accuracy: 65.40%, F1: 0.6293, Precision: 0.6422, Recall: 0.6540
# Fold 2, Epoch 25: Loss 2.2258, Accuracy 0.1549
# Fold 2, Epoch 50: Loss 1.1807, Accuracy 0.5496
# Fold 2, Epoch 75: Loss 1.0042, Accuracy 0.6084
# Fold 2, Epoch 100: Loss 0.9376, Accuracy 0.6382
# Fold 2 - Accuracy: 62.70%, F1: 0.6090, Precision: 0.6120, Recall: 0.6270
# Fold 3, Epoch 25: Loss 1.6674, Accuracy 0.3659
# Fold 3, Epoch 50: Loss 1.5087, Accuracy 0.4155
# Fold 3, Epoch 75: Loss 1.0717, Accuracy 0.5899
# Fold 3, Epoch 100: Loss 0.9841, Accuracy 0.6118
# Fold 3 - Accuracy: 61.60%, F1: 0.5996, Precision: 0.5973, Recall: 0.6160
# Fold 4, Epoch 25: Loss 1.6503, Accuracy 0.3620
# Fold 4, Epoch 50: Loss 1.5226, Accuracy 0.3982
# Fold 4, Epoch 75: Loss 1.1008, Accuracy 0.5807
# Fold 4, Epoch 100: Loss 0.9787, Accuracy 0.6210
# Fold 4 - Accuracy: 61.10%, F1: 0.5824, Precision: 0.6039, Recall: 0.6110
# Fold 5, Epoch 25: Loss 1.6030, Accuracy 0.3942
# Fold 5, Epoch 50: Loss 1.0043, Accuracy 0.6072
# Fold 5, Epoch 75: Loss 1.1783, Accuracy 0.5427
# Fold 5, Epoch 100: Loss 0.8661, Accuracy 0.6595
# Fold 5 - Accuracy: 67.75%, F1: 0.6661, Precision: 0.6721, Recall: 0.6775

# === Cross-Validation Summary ===
# Average Accuracy: 63.71%
# Average F1: 0.6173
# Average Precision: 0.6255
# Average Recall: 0.6371


# Fold 1, Epoch 25: Loss 2.2187, Accuracy 0.1649
# Fold 1, Epoch 50: Loss 1.0738, Accuracy 0.5887
# Fold 1, Epoch 75: Loss 0.9396, Accuracy 0.6354
# Fold 1, Epoch 100: Loss 0.8553, Accuracy 0.6641
# Fold 1 - Accuracy: 67.65%, F1: 0.6509, Precision: 0.6657, Recall: 0.6765
# Fold 2, Epoch 25: Loss 1.6331, Accuracy 0.3765
# Fold 2, Epoch 50: Loss 1.2527, Accuracy 0.5354
# Fold 2, Epoch 75: Loss 1.0139, Accuracy 0.6011
# Fold 2, Epoch 100: Loss 0.9680, Accuracy 0.6255
# Fold 2 - Accuracy: 62.65%, F1: 0.6161, Precision: 0.6295, Recall: 0.6265
# Fold 3, Epoch 25: Loss 2.2879, Accuracy 0.1344
# Fold 3, Epoch 50: Loss 1.5313, Accuracy 0.4121
# Fold 3, Epoch 75: Loss 1.1093, Accuracy 0.5671
# Fold 3, Epoch 100: Loss 0.9980, Accuracy 0.6140
# Fold 3 - Accuracy: 58.55%, F1: 0.5837, Precision: 0.5914, Recall: 0.5855
# Fold 4, Epoch 25: Loss 1.6473, Accuracy 0.3684
# Fold 4, Epoch 50: Loss 1.2688, Accuracy 0.5215
# Fold 4, Epoch 75: Loss 1.0906, Accuracy 0.5811
# Fold 4, Epoch 100: Loss 1.0256, Accuracy 0.6011
# Fold 4 - Accuracy: 60.40%, F1: 0.5844, Precision: 0.6154, Recall: 0.6040
# Fold 5, Epoch 25: Loss 1.6226, Accuracy 0.3718
# Fold 5, Epoch 50: Loss 1.0748, Accuracy 0.5859
# Fold 5, Epoch 75: Loss 0.9662, Accuracy 0.6268
# Fold 5, Epoch 100: Loss 0.9017, Accuracy 0.6478
# Fold 5 - Accuracy: 67.65%, F1: 0.6651, Precision: 0.6709, Recall: 0.6765
#
# === Cross-Validation Summary ===
# Average Accuracy: 63.38%
# Average F1: 0.6200
# Average Precision: 0.6346
# Average Recall: 0.6338

#primo è il solito, secondo è la cnn giusta, terzo è la lstm con drop, quarto lstm senza drop

# Fold 1, Epoch 25: Loss 2.2129, Accuracy 0.1663
# Fold 1, Epoch 50: Loss 1.0462, Accuracy 0.5956
# Fold 1, Epoch 75: Loss 0.9922, Accuracy 0.6165
# Fold 1, Epoch 100: Loss 0.9391, Accuracy 0.6294
# Fold 1 - Confusion Matrix:
# [[108   1  15   2  23   0   0   5  32  22]
#  [  1 111  11   0   5   0  29  34   2   9]
#  [ 40   6  82   0   3   0   8  10  35   8]
#  [  3   0   0 186   4   0   0   0   0   8]
#  [  9   3   4   3 168   0   0   8   2  12]
#  [  0   3   0   0   0 183   0   0   0   0]
#  [  0  12  18   0   0   0 176   5   0   0]
#  [  3  10   1   0  12   0   0 171   2   5]
#  [ 74   4  30   4  21   0   0  15  39  25]
#  [  9  31  10   7  27   1   5  21   9  55]]
# Fold 1 - Accuracy: 63.95%, F1: 0.6258, Precision: 0.6228, Recall: 0.6395

# Fold 2, Epoch 25: Loss 1.4416, Accuracy 0.4626
# Fold 2, Epoch 50: Loss 0.9980, Accuracy 0.6181
# Fold 2, Epoch 75: Loss 0.8984, Accuracy 0.6488
# Fold 2, Epoch 100: Loss 0.8081, Accuracy 0.6795
# Fold 2 - Confusion Matrix:
# [[101   2  44   0   7   0   0   1  44   7]
#  [  1 114  10   0   5   1  25  32   2  19]
#  [ 17   9 132   0   1   0   7   3  33   0]
#  [  7   0   0 185   5   0   0   0   0   3]
#  [ 21   0   1   4 174   0   0  11   5  13]
#  [  0   5   0   0   0 197   0   0   0   1]
#  [  0  18  11   0   0   0 132   1   0   0]
#  [  2   5  16   0   3   0   2 153   1   2]
#  [ 70   2  46   1   8   0   0  12  48  15]
#  [ 24  29  19   8   7   2   4  10  18  82]]
# Fold 2 - Accuracy: 65.90%, F1: 0.6547, Precision: 0.6604, Recall: 0.6590

# Fold 3, Epoch 25: Loss 1.0890, Accuracy 0.5816
# Fold 3, Epoch 50: Loss 0.8939, Accuracy 0.6501
# Fold 3, Epoch 75: Loss 0.8144, Accuracy 0.6774
# Fold 3, Epoch 100: Loss 0.7687, Accuracy 0.6923
# Fold 3 - Confusion Matrix:
# [[106   1  36   5  17   1   0   3  33  13]
#  [  1  90  11   0   2   3  45  20   3  24]
#  [ 26   4 137   0   2   0  15   8   3   9]
#  [  0   0   0 183   0   2   0   0   0   4]
#  [ 10   1   2   6 165   0   0   2   5   6]
#  [  0   3   0   0   0 205   1   0   0   3]
#  [  0   5  12   0   0   0 187   7   0   6]
#  [  1   8   7   0   8   0   0 152   0   5]
#  [ 69   2  44   2  16   0   1   7  30  24]
#  [ 19  19  12  13  13   1   2   7  11  94]]
# Fold 3 - Accuracy: 67.45%, F1: 0.6579, Precision: 0.6579, Recall: 0.6745

# Fold 4, Epoch 25: Loss 1.5514, Accuracy 0.3934
# Fold 4, Epoch 50: Loss 0.9728, Accuracy 0.6189
# Fold 4, Epoch 75: Loss 0.8384, Accuracy 0.6654
# Fold 4, Epoch 100: Loss 0.8082, Accuracy 0.6803
# Fold 4 - Confusion Matrix:
# [[115   1  25   0  11   0   1   1  11   4]
#  [  2 130   8   0   2   1  14  24   2   9]
#  [ 62   8 105   0   1   0  14  13   7   3]
#  [  4   0   0 208   3   0   0   0   0   1]
#  [ 22   0   0   4 138   0   0   1   1   2]
#  [  0   1   0   1   0 194   0   0   0   3]
#  [  0  27   6   0   0   0 170   6   0   1]
#  [  5  24   1   0  19   0   2 157   1   3]
#  [106   3  26   0  21   0   0   4  21   9]
#  [ 33  34  19   5  18   4   1  10  13  94]]
# Fold 4 - Accuracy: 66.60%, F1: 0.6548, Precision: 0.6811, Recall: 0.6660

# Fold 5, Epoch 25: Loss 2.2171, Accuracy 0.1590
# Fold 5, Epoch 50: Loss 1.4795, Accuracy 0.4385
# Fold 5, Epoch 75: Loss 1.0334, Accuracy 0.5951
# Fold 5, Epoch 100: Loss 0.9146, Accuracy 0.6445
# Fold 5 - Confusion Matrix:
# [[136   0  16   2  13   0   0   4  25   6]
#  [  2 107   4   0   5   1  28  26   1  24]
#  [ 48   5  87   1   4   0  14   8  20   2]
#  [  1   0   1 187   2   0   0   0   0   3]
#  [ 14   0   0   8 157   0   0   7   1  10]
#  [  0   0   0   1   0 199   0   0   0   0]
#  [  0   6  17   0   0   2 171   3   0   1]
#  [  3  13   6   0   8   0   0 185   1   3]
#  [ 95   0  10   2  28   0   0  12  35  19]
#  [ 12  20   8   9  24   5   2  15  20  85]]
# Fold 5 - Accuracy: 67.45%, F1: 0.6594, Precision: 0.6633, Recall: 0.6745

# === Cross-Validation Summary ===
# Average Accuracy: 66.27%
# Average F1: 0.6505
# Average Precision: 0.6571
# Average Recall: 0.6627

# Fold 1, Epoch 25: Loss 1.5860, Accuracy 0.3806
# Fold 1, Epoch 50: Loss 0.8731, Accuracy 0.6625
# Fold 1, Epoch 75: Loss 0.7853, Accuracy 0.6920
# Fold 1, Epoch 100: Loss 0.7499, Accuracy 0.7051
# Fold 1 - Confusion Matrix:
# [[104   2  38   3  14   0   0   2  36   9]
#  [  2 122   9   0   5   2  36  17   0   9]
#  [ 23   8 126   0   3   0  13   1  15   3]
#  [  2   0   0 191   2   0   0   0   2   4]
#  [ 16   1   2   3 174   0   0   4   5   4]
#  [  0   1   0   0   0 185   0   0   0   0]
#  [  0   3  10   0   0   0 196   2   0   0]
#  [  4  13   6   0   8   0   2 163   3   5]
#  [ 53   7  59   2  16   0   1   8  42  24]
#  [ 17  30   9   7  16   3   8  10   9  66]]
# Fold 1 - Accuracy: 68.45%, F1: 0.6702, Precision: 0.6691, Recall: 0.6845

# Fold 2, Epoch 25: Loss 1.4983, Accuracy 0.4071
# Fold 2, Epoch 50: Loss 0.9549, Accuracy 0.6316
# Fold 2, Epoch 75: Loss 0.8402, Accuracy 0.6650
# Fold 2, Epoch 100: Loss 0.8055, Accuracy 0.6793
# Fold 2 - Confusion Matrix:
# [[147   1  25   1   8   0   0   1  11  12]
#  [  1 107   5   0   7   3  34  30   0  22]
#  [ 63   9 104   0   0   0  14   1   7   4]
#  [  2   0   0 195   0   1   0   0   0   2]
#  [ 14   0   2   6 194   0   0   4   1   8]
#  [  0   2   0   0   0 200   0   0   0   1]
#  [  1   5   8   0   0   0 148   0   0   0]
#  [  5   9   7   0   6   0   5 150   1   1]
#  [115   3  33   3  17   0   0   9   9  13]
#  [ 25  24  18   9  22   5   6   5   9  80]]
# Fold 2 - Accuracy: 66.70%, F1: 0.6410, Precision: 0.6458, Recall: 0.6670

# Fold 3, Epoch 25: Loss 1.3402, Accuracy 0.4908
# Fold 3, Epoch 50: Loss 0.8124, Accuracy 0.6860
# Fold 3, Epoch 75: Loss 0.7707, Accuracy 0.6974
# Fold 3, Epoch 100: Loss 0.7471, Accuracy 0.7049
# Fold 3 - Confusion Matrix:
# [[108   1  29   3  14   1   1   2  39  17]
#  [  1 129   6   0   2   2  23  20   2  14]
#  [ 29  11 116   0   1   0  11   3  27   6]
#  [  1   0   0 183   0   1   0   0   0   4]
#  [ 11   0   0   4 158   0   0   1  11  12]
#  [  0   6   0   0   0 205   1   0   0   0]
#  [  0  22  13   0   0   0 176   3   0   3]
#  [  2   7   6   0   5   0   0 150   4   7]
#  [ 62   3  36   1  13   0   0   4  51  25]
#  [ 18  30   5  11  14   2   3   6  14  88]]
# Fold 3 - Accuracy: 68.20%, F1: 0.6767, Precision: 0.6733, Recall: 0.6820

# Fold 4, Epoch 25: Loss 1.5167, Accuracy 0.4120
# Fold 4, Epoch 50: Loss 1.4938, Accuracy 0.4143
# Fold 4, Epoch 75: Loss 0.9898, Accuracy 0.6174
# Fold 4, Epoch 100: Loss 0.8471, Accuracy 0.6660
# Fold 4 - Confusion Matrix:
# [[ 66   0  44   0  12   0   1   0  37   9]
#  [  0 129   8   0   1   1  17  26   1   9]
#  [ 17   8 157   0   0   0   8   3  16   4]
#  [  0   0   0 209   4   0   0   0   1   2]
#  [  9   1   2   3 134   0   0   0  12   7]
#  [  0   1   0   1   0 195   0   0   0   2]
#  [  0  12  18   0   0   0 177   1   0   2]
#  [  0  16  20   0   9   0   2 157   2   6]
#  [ 45   8  53   0  13   0   0   3  52  16]
#  [ 12  47  16   6  16   2   2  10  15 105]]
# Fold 4 - Accuracy: 69.05%, F1: 0.6844, Precision: 0.6906, Recall: 0.6905

# Fold 5, Epoch 25: Loss 1.5064, Accuracy 0.4042
# Fold 5, Epoch 50: Loss 0.8412, Accuracy 0.6727
# Fold 5, Epoch 75: Loss 0.7894, Accuracy 0.6881
# Fold 5, Epoch 100: Loss 0.7615, Accuracy 0.6995
# Fold 5 - Confusion Matrix:
# [[142   0  42   0   6   0   0   1   6   5]
#  [  1 129  13   0   5   1  15  10   4  20]
#  [ 41   6 125   0   2   0   5   3   3   4]
#  [  2   0   1 188   1   0   0   0   0   2]
#  [ 18   2   0   8 160   0   0   3   3   3]
#  [  0  10   0   0   0 188   1   0   0   1]
#  [  1  13  25   0   0   0 157   1   0   3]
#  [  6  33   8   0  12   0   1 155   2   2]
#  [108   1  36   1  13   0   0   5  17  20]
#  [ 31  22  11   8  17   3   3   7   9  89]]
# Fold 5 - Accuracy: 67.50%, F1: 0.6610, Precision: 0.6818, Recall: 0.6750

# === Cross-Validation Summary ===
# Average Accuracy: 67.98%
# Average F1: 0.6667
# Average Precision: 0.6721
# Average Recall: 0.6798
