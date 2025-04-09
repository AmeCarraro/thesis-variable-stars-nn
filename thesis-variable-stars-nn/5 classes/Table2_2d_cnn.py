import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Use non-interactive backend for saving plots
import matplotlib
matplotlib.use('Agg')

# Paths
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"
save_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\histograms\\2d_cnn\\"
composite_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\histograms\\2d_cnn\\composite_histograms"
os.makedirs(save_path, exist_ok=True)
os.makedirs(composite_path, exist_ok=True)

# Create 2D histograms (dm-dt) from pairwise differences
def calculate_pairwise_differences(data, bins_dm, bins_dt):
    histograms = []
    for index1, row1 in data.iterrows():
        dm = []
        dt = []
        for index2, row2 in data.iterrows():
            if index1 != index2:
                # Compute pairwise brightness and period differences
                delta_m_g = abs(row1['Amp_g'] - row2['Amp_g'])
                delta_m_r = abs(row1['Amp_r'] - row2['Amp_r'])
                dm.extend([delta_m_g, delta_m_r])
                delta_t = abs(row1['Per'] - row2['Per'])
                dt.append(delta_t)
        # Create and normalize 2D histogram
        min_length = min(len(dm), len(dt))
        dm = dm[:min_length]
        dt = dt[:min_length]
        dmdt_grid, _, _ = np.histogram2d(dt, dm, bins=[bins_dt, bins_dm])
        max_value = dmdt_grid.max()
        if max_value > 0:
            dmdt_grid = (dmdt_grid / max_value) * 255
        dmdt_grid = resize(dmdt_grid, (90, 53), anti_aliasing=True)
        histograms.append(dmdt_grid)
    return histograms

# Process dataset, save histograms and composite plots
def calculate_and_save_histograms(df, bins_dm, bins_dt, fold_num, dataset_type):
    histograms = []
    labels = []
    for star_type in df['Type'].unique():
        subset = df[df['Type'] == star_type]
        dmdt_histograms = calculate_pairwise_differences(subset, bins_dm=bins_dm, bins_dt=bins_dt)
        composite_grid = np.zeros((90, 53))
        for idx, dmdt_grid in enumerate(dmdt_histograms):
            histograms.append(dmdt_grid)
            labels.append(star_type)
            # Save first 10 individual histograms per type
            if idx < 10:
                plt.imshow(dmdt_grid, cmap='viridis', origin='lower', interpolation='nearest')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(save_path, f'fold{fold_num}_{dataset_type}_histogram_{star_type}_{idx}.png'))
                plt.close()
            composite_grid += dmdt_grid
        # Save composite histogram (average of first 10)
        composite_grid /= min(len(dmdt_histograms), 10)
        plt.imshow(composite_grid, cmap='viridis', origin='lower', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(composite_path, f'fold{fold_num}_{dataset_type}_composite_dmdt_{star_type}.png'))
        plt.close()
    return histograms, labels

# Define binning for the 2D histograms
bins_dm = [0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8]
bins_dt = [1/145, 2/145, 4/145, 1/25, 2/25, 3/25, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 10, 20, 30, 60, 90, 120, 250, 600, 960, 2000, 4000]

# 2D CNN model for image classification
class My2DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(My2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5)
        self.fc1 = nn.Linear(256 * 37 * 18, 512)
        self.dropout_fc = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
num_epochs = 100

# Metrics containers
fold_results = []
all_f1_scores = []
all_precision = []
all_recall = []
all_accuracy = []

# Cross-validation loop
for i in range(1, 6):
    # Load fold data
    train_df = pd.read_csv(Path(f'{data_path}5_cnn_train_fold_{i}.txt'), sep=' ')
    test_df = pd.read_csv(Path(f'{data_path}5_cnn_test_fold_{i}.txt'), sep=' ')
    
    # Generate and save 2D histograms for train and test sets
    histograms_train, labels_train = calculate_and_save_histograms(train_df, bins_dm, bins_dt, i, 'train')
    histograms_test, labels_test = calculate_and_save_histograms(test_df, bins_dm, bins_dt, i, 'test')
    
    # Convert to NumPy arrays
    X_train = np.array(histograms_train)
    y_train = np.array(labels_train)
    X_test = np.array(histograms_test)
    y_test = np.array(labels_test)

    # Encode class labels
    y_train = pd.factorize(y_train)[0]
    y_test = pd.factorize(y_test)[0]

    # Convert grayscale images to 3 channels (RGB)
    X_train = np.repeat(X_train[:, np.newaxis, :, :], 3, axis=1)
    X_test = np.repeat(X_test[:, np.newaxis, :, :], 3, axis=1)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Create data loaders
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataset_test = TensorDataset(X_test_tensor, y_test_tensor)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

    # Initialize model, loss function and optimizer
    model = My2DCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
        train_accuracy = correct_predictions / total_predictions
        if (epoch + 1) % 10 == 0:
            print(f"Fold {i}, Epoch {epoch+1}, Loss: {running_loss / len(dataloader_train):.4f}, Train Acc: {train_accuracy:.4f}")

    # Evaluation loop
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    test_loss = 0.0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    test_loss /= len(dataloader_test)
    test_accuracy = correct_predictions / total_predictions
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"Fold {i} - Confusion Matrix:")
    print(cm)
    acc = accuracy_score(all_labels, all_predictions)
    prec = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    print(f"Fold {i} - Test Loss: {test_loss:.4f}, Test Accuracy: {acc*100:.2f}%, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

    # Store results
    fold_results.append((test_loss, acc, prec, rec, f1))
    all_accuracy.append(acc)
    all_precision.append(prec)
    all_recall.append(rec)
    all_f1_scores.append(f1)

# Summary of all folds
avg_loss = np.mean([result[0] for result in fold_results])
avg_accuracy = np.mean(all_accuracy)
avg_precision = np.mean(all_precision)
avg_recall = np.mean(all_recall)
avg_f1_score = np.mean([result[4] for result in fold_results])

print(f"\nAverage Cross-Validation Loss: {avg_loss:.4f}")
print(f"Average Cross-Validation Accuracy: {avg_accuracy*100:.2f}%")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1_score:.4f}")

# Print per-fold results
for i, result in enumerate(fold_results):
    loss, acc, prec, rec, f1 = result
    print(f"Fold {i+1} - Loss: {loss:.4f}, Accuracy: {acc*100:.2f}%, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")





#Fold 1 - Test Loss: 0.8298, Test Accuracy: 88.20%, Precision: 0.8213, Recall: 0.8820, F1 Score: 0.8436
#Fold 2 - Test Loss: 0.4550, Test Accuracy: 91.50%, Precision: 0.9230, Recall: 0.9150, F1 Score: 0.9109
#Fold 3 - Test Loss: 0.1123, Test Accuracy: 97.30%, Precision: 0.9752, Recall: 0.9730, F1 Score: 0.9730
#Fold 4 - Test Loss: 0.0670, Test Accuracy: 97.90%, Precision: 0.9794, Recall: 0.9790, F1 Score: 0.9790
#Fold 5 - Test Loss: 0.0685, Test Accuracy: 98.30%, Precision: 0.9832, Recall: 0.9830, F1 Score: 0.9829

#Average Cross-Validation Loss: 0.3065
#Average Cross-Validation Accuracy: 94.64%
#Average Precision: 0.9365
#Average Recall: 0.9464
#Average F1 Score: 0.9379
#Fold 1 - Loss: 0.8298, Accuracy: 88.20%, Precision: 0.8213, Recall: 0.8820, F1 Score: 0.8436
#Fold 2 - Loss: 0.4550, Accuracy: 91.50%, Precision: 0.9230, Recall: 0.9150, F1 Score: 0.9109
#Fold 3 - Loss: 0.1123, Accuracy: 97.30%, Precision: 0.9752, Recall: 0.9730, F1 Score: 0.9730
#Fold 4 - Loss: 0.0670, Accuracy: 97.90%, Precision: 0.9794, Recall: 0.9790, F1 Score: 0.9790
#Fold 5 - Loss: 0.0685, Accuracy: 98.30%, Precision: 0.9832, Recall: 0.9830, F1 Score: 0.9829

#Average Cross-Validation Loss: 0.3858
#Average Cross-Validation Accuracy: 93.05%
#Average Precision: 0.9428
#Average Recall: 0.9305
#Average F1 Score: 0.9197
#Fold 1 - Loss: 0.7737, Accuracy: 88.55%, Precision: 0.9169, Recall: 0.8855, F1 Score: 0.8437
#Fold 2 - Loss: 0.5131, Accuracy: 90.65%, Precision: 0.9157, Recall: 0.9065, F1 Score: 0.9044
#Fold 3 - Loss: 0.1390, Accuracy: 96.90%, Precision: 0.9715, Recall: 0.9690, F1 Score: 0.9688
#Fold 4 - Loss: 0.4194, Accuracy: 91.35%, Precision: 0.9311, Recall: 0.9135, F1 Score: 0.9037
#Fold 5 - Loss: 0.0840, Accuracy: 97.80%, Precision: 0.9786, Recall: 0.9780, F1 Score: 0.9779

