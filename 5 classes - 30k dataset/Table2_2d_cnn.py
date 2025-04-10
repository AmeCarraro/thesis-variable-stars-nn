import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib
matplotlib.use('Agg')

# ---- Simple logging utility ----
log_filename = "output_log.txt"
log_file = open(log_filename, "w")
def log_print(message):
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

# ---- Calculate pairwise 2D histograms ----
def calculate_pairwise_differences(data, bins_dm, bins_dt):
    histograms = []
    for index1, row1 in data.iterrows():
        dm, dt = [], []
        for index2, row2 in data.iterrows():
            if index1 != index2:
                delta_m_g = abs(row1['Amp_g'] - row2['Amp_g'])
                delta_m_r = abs(row1['Amp_r'] - row2['Amp_r'])
                dm.extend([delta_m_g, delta_m_r])
                dt.append(abs(row1['Per'] - row2['Per']))
        min_len = min(len(dm), len(dt))
        dm, dt = dm[:min_len], dt[:min_len]
        grid, _, _ = np.histogram2d(dt, dm, bins=[bins_dt, bins_dm])
        if grid.max() > 0:
            grid = (grid / grid.max()) * 255
        grid = resize(grid, (90, 53), anti_aliasing=True)
        histograms.append(grid)
    return histograms

# ---- Calculate and log histograms per star type ----
def calculate_and_save_histograms(df, bins_dm, bins_dt, fold_num, dataset_type):
    histograms, labels = [], []
    for star_type in df['Type'].unique():
        start = time.time()
        subset = df[df['Type'] == star_type]
        dmdt_histograms = calculate_pairwise_differences(subset, bins_dm, bins_dt)
        composite = np.zeros((90, 53))
        for dmdt_grid in dmdt_histograms:
            histograms.append(dmdt_grid)
            labels.append(star_type)
            composite += dmdt_grid
        composite /= min(len(dmdt_histograms), 10)
        elapsed = time.time() - start
        log_print(f"Fold {fold_num}, {dataset_type}: '{star_type}' computed in {elapsed:.2f} sec.")
    return histograms, labels

# ---- Bin definitions ----
bins_dm = [0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8]
bins_dt = [1/145, 2/145, 4/145, 1/25, 2/25, 3/25, 1.5, 2.5, 3.5, 4.5, 5.5, 7,
           10, 20, 30, 60, 90, 120, 250, 600, 960, 2000, 4000]

# ---- 2D CNN Definition ----
class My2DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(My2DCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)   # (B, 64, 90, 53)
        self.pool = nn.MaxPool2d(2, 2)                  # -> (B, 64, 45, 26)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=0)   # -> (B, 128, 41, 22)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=0)  # -> (B, 256, 37, 18)
        self.fc1 = nn.Linear(256 * 37 * 18, 512)
        self.dropout_fc = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)

# ---- Main Training & Testing Loop ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr, num_epochs = 0.0002, 100
fold_results = []
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"

# Loop over selected folds (4 and 5 in this example)
for i in range(4, 6):
    train_df = pd.read_csv(Path(f'{data_path}inf_cnn_train_fold_{i}.txt'), sep=' ')
    test_df = pd.read_csv(Path(f'{data_path}inf_cnn_test_fold_{i}.txt'), sep=' ')
    
    # Filter to 5 classes
    classes = ['CEP', 'DSCT', 'EA', 'RR', 'LPV']
    train_df = train_df[train_df['Type'].isin(classes)]
    test_df = test_df[test_df['Type'].isin(classes)]
    
    hist_train, labels_train = calculate_and_save_histograms(train_df, bins_dm, bins_dt, i, 'train')
    hist_test, labels_test = calculate_and_save_histograms(test_df, bins_dm, bins_dt, i, 'test')
    
    X_train = np.array(hist_train)
    y_train = np.array(labels_train)
    X_test = np.array(hist_test)
    y_test = np.array(labels_test)
    y_train = pd.factorize(y_train)[0]
    y_test = pd.factorize(y_test)[0]
    
    X_train = np.repeat(X_train[:, None, :, :], 3, axis=1)
    X_test = np.repeat(X_test[:, None, :, :], 3, axis=1)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)
    
    model = My2DCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0; correct = 0; total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        if (epoch+1) % 10 == 0:
            log_print(f"Fold {i}, Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {correct/total:.4f}")
    
    # Testing phase
    model.eval()
    correct = 0; total = 0; test_loss = 0; all_preds = []; all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss /= len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    log_print(f"Fold {i} - Test Loss: {test_loss:.4f}, Acc: {acc*100:.2f}%, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    fold_results.append((test_loss, acc, prec, rec, f1))

# Cross-validation summary
avg_loss = np.mean([r[0] for r in fold_results])
avg_acc = np.mean([r[1] for r in fold_results])
avg_prec = np.mean([r[2] for r in fold_results])
avg_rec = np.mean([r[3] for r in fold_results])
avg_f1 = np.mean([r[4] for r in fold_results])
log_print(f"\nAvg Loss: {avg_loss:.4f}")
log_print(f"Avg Acc: {avg_acc*100:.2f}%")
log_print(f"Avg Prec: {avg_prec:.4f}")
log_print(f"Avg Rec: {avg_rec:.4f}")
log_print(f"Avg F1: {avg_f1:.4f}")

log_file.close()




#Fold 1, dataset train: Classe 'CEP' calcolata in 26.69 secondi.
#Fold 1, dataset train: Classe 'DSCT' calcolata in 5066.47 secondi.
#Fold 1, dataset train: Classe 'EA' calcolata in 22571.07 secondi.
#Fold 1, dataset train: Classe 'LPV' calcolata in 28126.64 secondi.
#Fold 1, dataset train: Classe 'RR' calcolata in 12343.76 secondi.
#Fold 1, dataset test: Classe 'CEP' calcolata in 1.91 secondi.
#Fold 1, dataset test: Classe 'DSCT' calcolata in 288.26 secondi.
#Fold 1, dataset test: Classe 'EA' calcolata in 1428.25 secondi.
#Fold 1, dataset test: Classe 'LPV' calcolata in 1450.77 secondi.
#Fold 1, dataset test: Classe 'RR' calcolata in 437.47 secondi.
#Fold 1, Epoch 10, Loss: 0.0008, Train Acc: 0.9997
#Fold 1, Epoch 20, Loss: 0.0007, Train Acc: 0.9998
#Fold 1, Epoch 30, Loss: 0.0007, Train Acc: 0.9998
#Fold 1, Epoch 40, Loss: 0.0002, Train Acc: 0.9999
#Fold 1, Epoch 50, Loss: 0.0001, Train Acc: 0.9999
#Fold 1, Epoch 60, Loss: 0.0004, Train Acc: 0.9999
#Fold 1, Epoch 70, Loss: 0.0002, Train Acc: 1.0000
#Fold 1, Epoch 80, Loss: 0.0011, Train Acc: 0.9999
#Fold 1, Epoch 90, Loss: 0.0003, Train Acc: 0.9999
#Fold 1, Epoch 100, Loss: 0.0003, Train Acc: 0.9999
#Fold 1 - Test Loss: 0.0011, Test Accuracy: 99.98%, Precision: 0.9998, Recall: 0.9998, F1 Score: 0.9998


# Fold 1, dataset train: Classe 'CEP' calcolata in 26.69 secondi.
# Fold 1, dataset train: Classe 'DSCT' calcolata in 5066.47 secondi.
# Fold 1, dataset train: Classe 'EA' calcolata in 22571.07 secondi.
# Fold 1, dataset train: Classe 'LPV' calcolata in 28126.64 secondi.
# Fold 1, dataset train: Classe 'RR' calcolata in 12343.76 secondi.
# Fold 1, dataset test: Classe 'CEP' calcolata in 1.91 secondi.
# Fold 1, dataset test: Classe 'DSCT' calcolata in 288.26 secondi.
# Fold 1, dataset test: Classe 'EA' calcolata in 1428.25 secondi.
# Fold 1, dataset test: Classe 'LPV' calcolata in 1450.77 secondi.
# Fold 1, dataset test: Classe 'RR' calcolata in 437.47 secondi.
# Fold 1, Epoch 10, Loss: 0.0008, Train Acc: 0.9997
# Fold 1, Epoch 20, Loss: 0.0007, Train Acc: 0.9998
# Fold 1, Epoch 70, Loss: 0.0002, Train Acc: 1.0000
# Fold 1, Epoch 80, Loss: 0.0011, Train Acc: 0.9999
# Fold 1, Epoch 90, Loss: 0.0003, Train Acc: 0.9999
# Fold 1, Epoch 100, Loss: 0.0003, Train Acc: 0.9999
# Fold 1 - Test Loss: 0.0011, Test Accuracy: 99.98%, Precision: 0.9998, Recall: 0.9998, F1 Score: 0.9998

# Fold 2, dataset train: Classe 'CEP' calcolata in 26.66 secondi.
# Fold 2, dataset train: Classe 'DSCT' calcolata in 9210.25 secondi.
# Fold 2, dataset train: Classe 'EA' calcolata in 23670.29 secondi.
# Fold 2, dataset train: Classe 'LPV' calcolata in 27019.02 secondi.
# Fold 2, dataset train: Classe 'RR' calcolata in 12160.92 secondi.
# Fold 2, dataset test: Classe 'CEP' calcolata in 2.16 secondi.
# Fold 2, dataset test: Classe 'DSCT' calcolata in 273.79 secondi.
# Fold 2, dataset test: Classe 'EA' calcolata in 1446.40 secondi.
# Fold 2, dataset test: Classe 'LPV' calcolata in 1386.48 secondi.
# Fold 2, dataset test: Classe 'RR' calcolata in 488.61 secondi.
# Fold 2, Epoch 10, Loss: 0.0008, Train Acc: 0.9998
# Fold 2, Epoch 20, Loss: 0.0006, Train Acc: 0.9999
# Fold 2, Epoch 30, Loss: 0.0007, Train Acc: 0.9998
# Fold 2, Epoch 40, Loss: 0.0004, Train Acc: 0.9999
# Fold 2, Epoch 50, Loss: 0.0001, Train Acc: 0.9999
# Fold 2, Epoch 60, Loss: 0.0001, Train Acc: 1.0000
# Fold 2, Epoch 70, Loss: 0.0002, Train Acc: 0.9999
# Fold 2, Epoch 80, Loss: 0.0004, Train Acc: 0.9999
# Fold 2, Epoch 90, Loss: 0.0011, Train Acc: 0.9998
# Fold 2, Epoch 100, Loss: 0.0006, Train Acc: 0.9999
# Fold 2 - Test Loss: 0.0060, Test Accuracy: 99.96%, Precision: 0.9996, Recall: 0.9996, F1 Score: 0.9996

# Fold 3, dataset train: Classe 'CEP' calcolata in 34.42 secondi.
# Fold 3, dataset train: Classe 'DSCT' calcolata in 4544.38 secondi.
# Fold 3, dataset train: Classe 'EA' calcolata in 26149.20 secondi.
# Fold 3, dataset train: Classe 'LPV' calcolata in 23525.78 secondi.
# Fold 3, dataset train: Classe 'RR' calcolata in 14763.87 secondi.
# Fold 3, dataset test: Classe 'CEP' calcolata in 4.65 secondi.
# Fold 3, dataset test: Classe 'DSCT' calcolata in 651.91 secondi.
# Fold 3, dataset test: Classe 'EA' calcolata in 2701.78 secondi.
# Fold 3, dataset test: Classe 'LPV' calcolata in 3935.82 secondi.
# Fold 3, dataset test: Classe 'RR' calcolata in 926.62 secondi.
