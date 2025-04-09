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
import matplotlib
matplotlib.use('Agg')

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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

# Define file paths for data and directories to save histograms and composite images
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"
save_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\histograms\\2d_cnn\\"
composite_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\histograms\\2d_cnn\\composite_histograms"
os.makedirs(save_path, exist_ok=True)
os.makedirs(composite_path, exist_ok=True)

# -----------------------------------------------------------------------------
# Function to calculate pairwise 2D histogram differences for a given dataset.
def calculate_pairwise_differences(data, bins_dm, bins_dt):
    histograms = []
    # Iterate over each star (row) in the dataset
    for index1, row1 in data.iterrows():
        dm = []  # List to store differences in magnitude (from both g and r bands)
        dt = []  # List to store differences in period
        # Compare with all other stars to compute pairwise differences
        for index2, row2 in data.iterrows():
            if index1 != index2:
                delta_m_g = abs(row1['Amp_g'] - row2['Amp_g'])
                delta_m_r = abs(row1['Amp_r'] - row2['Amp_r'])
                dm.append(delta_m_g)
                dm.append(delta_m_r)
                delta_t = abs(row1['Per'] - row2['Per'])
                dt.append(delta_t)
        # Ensure both lists have the same length by truncating to minimum length
        min_length = min(len(dm), len(dt))
        dm = dm[:min_length]
        dt = dt[:min_length]
        # Compute a 2D histogram using the provided bin ranges for dt and dm
        dmdt_grid, _, _ = np.histogram2d(dt, dm, bins=[bins_dt, bins_dm])
        # Normalize and scale the histogram grid to the range [0, 255]
        max_value = dmdt_grid.max()
        if max_value > 0:
            dmdt_grid = (dmdt_grid / max_value) * 255
        # Resize the grid to fixed dimensions (90, 53)
        dmdt_grid = resize(dmdt_grid, (90, 53), anti_aliasing=True)
        histograms.append(dmdt_grid)
    return histograms

# -----------------------------------------------------------------------------
# Function to calculate histograms for each star type, save sample and composite images.
def calculate_and_save_histograms(df, bins_dm, bins_dt, fold_num, dataset_type):
    histograms = []
    labels = []
    # Process each unique star type separately
    for star_type in df['Type'].unique():
        subset = df[df['Type'] == star_type]
        dmdt_histograms = calculate_pairwise_differences(subset, bins_dm=bins_dm, bins_dt=bins_dt)
        composite_grid = np.zeros((90, 53))
        # For each histogram in the subset, store and save images for sample histograms
        for idx, dmdt_grid in enumerate(dmdt_histograms):
            histograms.append(dmdt_grid)
            labels.append(star_type)
            # Save up to 10 example histograms per star type
            if idx < 10:
                plt.imshow(dmdt_grid, cmap='viridis', origin='lower', interpolation='nearest')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(save_path, f'fold{fold_num}_{dataset_type}_histogram_{star_type}_{idx}.png'))
                plt.close()
            composite_grid += dmdt_grid
        # Create a composite histogram by averaging the first 10 histograms (or fewer if not available)
        composite_grid /= min(len(dmdt_histograms), 10)
        plt.imshow(composite_grid, cmap='viridis', origin='lower', interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(composite_path, f'fold{fold_num}_{dataset_type}_composite_dmdt_{star_type}.png'))
        plt.close()
    return histograms, labels

# -----------------------------------------------------------------------------
# Define bin edges for magnitude and time differences
bins_dm = [0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8]
bins_dt = [1/145, 2/145, 4/145, 1/25, 2/25, 3/25, 1.5, 2.5, 3.5, 4.5, 5.5, 7, 10, 20, 30, 60, 90, 120, 250, 600, 960, 2000, 4000]

# -----------------------------------------------------------------------------
# Define a simple 2D CNN model for classifying the histogram images.
class My2DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(My2DCNN, self).__init__()
        # First convolution layer converts 3-channel input to 64 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        # Max pooling reduces spatial dimensions by a factor of 2
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.1)
        # Additional convolution layers with larger kernel sizes
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=0)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=0)
        # Fully connected layers; the input size is determined by output feature map dimensions
        self.fc1 = nn.Linear(256 * 37 * 18, 512)
        self.dropout_fc = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply convolution, activation, pooling and dropout
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        # Flatten feature maps into a vector
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

# -----------------------------------------------------------------------------
# Set device and training hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
num_epochs = 100

# Initialize lists to accumulate metrics for each fold
fold_results = []
all_f1_scores = []
all_precision = []
all_recall = []
all_accuracy = []

# Loop over each of the 5 cross-validation folds
for i in range(1, 6):
    # Load training and testing CSV files for the current fold
    train_df = pd.read_csv(Path(f'{data_path}2d_cnn_train_fold_{i}.txt'), sep=' ')
    test_df = pd.read_csv(Path(f'{data_path}2d_cnn_test_fold_{i}.txt'), sep=' ')
    
    # Calculate and save histograms (and composite images) for the training set
    histograms_train, labels_train = calculate_and_save_histograms(train_df, bins_dm, bins_dt, i, 'train')
    # Calculate histograms for the test set
    histograms_test, labels_test = calculate_and_save_histograms(test_df, bins_dm, bins_dt, i, 'test')
    
    # Convert lists of histograms and labels to numpy arrays
    X_train = np.array(histograms_train)
    y_train = np.array(labels_train)
    X_test = np.array(histograms_test)
    y_test = np.array(labels_test)
    
    # Factorize the labels so they become numeric codes
    y_train = pd.factorize(y_train)[0]
    y_test = pd.factorize(y_test)[0]
    
    # Repeat the single-channel (grayscale) histogram across 3 channels (RGB) 
    X_train = np.repeat(X_train[:, np.newaxis, :, :], 3, axis=1)
    X_test = np.repeat(X_test[:, np.newaxis, :, :], 3, axis=1)
    
    # Convert the numpy arrays to torch tensors and move to the selected device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # Create datasets and data loaders for training and testing
    dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataset_test = TensorDataset(X_test_tensor, y_test_tensor)
    dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)
    
    # Initialize the 2D CNN model, loss function, and optimizer
    model = My2DCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop for the current fold
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
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Fold {i}, Epoch {epoch+1}, Loss: {running_loss / len(dataloader_train):.4f}, Train Acc: {train_accuracy:.4f}")
    
    # Evaluation phase for the current fold
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
    test_loss /= len(dataloader_test)
    test_accuracy = correct_predictions / total_predictions
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"Fold {i} - Confusion Matrix:")
    print(cm)
    
    # Compute evaluation metrics using sklearn
    acc = accuracy_score(all_labels, all_predictions)
    prec = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    print(f"Fold {i} - Test Loss: {test_loss:.4f}, Test Accuracy: {acc*100:.2f}%, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
    
    # Accumulate metrics for later averaging
    all_accuracy.append(acc)
    all_precision.append(prec)
    all_recall.append(rec)
    fold_results.append((test_loss, acc, prec, rec, f1))

# Compute and print average metrics over all folds
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

# Print detailed results for each fold
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

# Fold 1, Epoch 10, Loss: 0.0089, Train Acc: 0.9979
# Fold 1, Epoch 20, Loss: 0.0121, Train Acc: 0.9972
# Fold 1, Epoch 30, Loss: 0.0153, Train Acc: 0.9962
# Fold 1, Epoch 40, Loss: 0.0069, Train Acc: 0.9990
# Fold 1, Epoch 50, Loss: 0.0150, Train Acc: 0.9961
# Fold 1, Epoch 60, Loss: 0.0065, Train Acc: 0.9991
# Fold 1, Epoch 70, Loss: 0.0001, Train Acc: 1.0000
# Fold 1, Epoch 80, Loss: 0.0075, Train Acc: 0.9981
# Fold 1, Epoch 90, Loss: 0.0050, Train Acc: 0.9988
# Fold 1, Epoch 100, Loss: 0.0001, Train Acc: 1.0000
# Fold 1 - Test Loss: 1.2891, Test Accuracy: 87.55%, Precision: 0.8143, Recall: 0.8755, F1 Score: 0.8372

# Fold 2, Epoch 10, Loss: 0.0133, Train Acc: 0.9960
# Fold 2, Epoch 20, Loss: 0.0035, Train Acc: 0.9986
# Fold 2, Epoch 30, Loss: 0.0016, Train Acc: 0.9995
# Fold 2, Epoch 40, Loss: 0.0033, Train Acc: 0.9990
# Fold 2, Epoch 50, Loss: 0.0029, Train Acc: 0.9994
# Fold 2, Epoch 60, Loss: 0.0009, Train Acc: 0.9998
# Fold 2, Epoch 70, Loss: 0.0155, Train Acc: 0.9952
# Fold 2, Epoch 80, Loss: 0.0001, Train Acc: 1.0000
# Fold 2, Epoch 90, Loss: 0.0073, Train Acc: 0.9972
# Fold 2, Epoch 100, Loss: 0.0027, Train Acc: 0.9990
# Fold 2 - Test Loss: 1.2941, Test Accuracy: 82.30%, Precision: 0.8784, Recall: 0.8230, F1 Score: 0.8041

# Fold 3, Epoch 10, Loss: 0.0065, Train Acc: 0.9979
# Fold 3, Epoch 20, Loss: 0.0145, Train Acc: 0.9969
# Fold 3, Epoch 30, Loss: 0.0003, Train Acc: 1.0000
# Fold 3, Epoch 40, Loss: 0.0006, Train Acc: 0.9998
# Fold 3, Epoch 50, Loss: 0.0002, Train Acc: 1.0000
# Fold 3, Epoch 60, Loss: 0.0001, Train Acc: 1.0000
# Fold 3, Epoch 70, Loss: 0.0001, Train Acc: 1.0000
# Fold 3, Epoch 80, Loss: 0.0001, Train Acc: 1.0000
# Fold 3, Epoch 90, Loss: 0.0011, Train Acc: 0.9998
# Fold 3, Epoch 100, Loss: 0.0001, Train Acc: 1.0000
# Fold 3 - Test Loss: 0.3185, Test Accuracy: 94.40%, Precision: 0.9502, Recall: 0.9440, F1 Score: 0.9438

# Fold 4, Epoch 10, Loss: 0.0229, Train Acc: 0.9938
# Fold 4, Epoch 20, Loss: 0.0390, Train Acc: 0.9930
# Fold 4, Epoch 30, Loss: 0.0285, Train Acc: 0.9936
# Fold 4, Epoch 40, Loss: 0.0034, Train Acc: 0.9989
# Fold 4, Epoch 50, Loss: 0.0137, Train Acc: 0.9960
# Fold 4, Epoch 60, Loss: 0.0141, Train Acc: 0.9962
# Fold 4, Epoch 70, Loss: 0.0012, Train Acc: 0.9999
# Fold 4, Epoch 80, Loss: 0.0009, Train Acc: 0.9998
# Fold 4, Epoch 90, Loss: 0.0000, Train Acc: 1.0000
# Fold 4, Epoch 100, Loss: 0.0068, Train Acc: 0.9984
# Fold 4 - Test Loss: 0.5341, Test Accuracy: 89.80%, Precision: 0.9156, Recall: 0.8980, F1 Score: 0.8981

# Fold 5, Epoch 10, Loss: 0.0047, Train Acc: 0.9982
# Fold 5, Epoch 20, Loss: 0.0206, Train Acc: 0.9940
# Fold 5, Epoch 30, Loss: 0.0067, Train Acc: 0.9979
# Fold 5, Epoch 40, Loss: 0.0080, Train Acc: 0.9979
# Fold 5, Epoch 50, Loss: 0.0005, Train Acc: 0.9999
# Fold 5, Epoch 60, Loss: 0.0014, Train Acc: 0.9996
# Fold 5, Epoch 70, Loss: 0.0000, Train Acc: 1.0000
# Fold 5, Epoch 80, Loss: 0.0008, Train Acc: 0.9996
# Fold 5, Epoch 90, Loss: 0.0027, Train Acc: 0.9995
# Fold 5, Epoch 100, Loss: 0.0125, Train Acc: 0.9971
# Fold 5 - Test Loss: 0.4486, Test Accuracy: 89.80%, Precision: 0.8896, Recall: 0.8980, F1 Score: 0.8833

# Average Cross-Validation Loss: 0.7769
# Average Cross-Validation Accuracy: 88.77%
# Average Precision: 0.8896
# Average Recall: 0.8877
# Average F1 Score: 0.8733



# Fold 1, Epoch 10, Loss: 0.0116, Train Acc: 0.9958
# Fold 1, Epoch 20, Loss: 0.0054, Train Acc: 0.9986
# Fold 1, Epoch 30, Loss: 0.0058, Train Acc: 0.9984
# Fold 1, Epoch 40, Loss: 0.0081, Train Acc: 0.9980
# Fold 1, Epoch 50, Loss: 0.0032, Train Acc: 0.9996
# Fold 1, Epoch 60, Loss: 0.0004, Train Acc: 1.0000
# Fold 1, Epoch 70, Loss: 0.0010, Train Acc: 0.9995
# Fold 1, Epoch 80, Loss: 0.0093, Train Acc: 0.9972
# Fold 1, Epoch 90, Loss: 0.0020, Train Acc: 0.9992
# Fold 1, Epoch 100, Loss: 0.0001, Train Acc: 1.0000
# Fold 1 - Confusion Matrix:
# [[208   0   0   0   0   0   0   0   0   0]
#  [  0 182   0   0   3   0   0   0  17   0]
#  [  0   0 179   0   0   0   0  13   0   0]
#  [  0   0   0 200   1   0   0   0   0   0]
#  [  0   0  21   0   0   0 143  45   0   0]
#  [  0   0   0   0   0 186   0   0   0   0]
#  [  0   0   0   0   0   0 211   0   0   0]
#  [  0   0   5   0   0   0   0 199   0   0]
#  [ 64   0   0   0   0   0   0   0 148   0]
#  [  0   0   0   0   0   1   0   0   0 174]]
# Fold 1 - Test Loss: 1.2946, Test Accuracy: 84.35%, Precision: 0.7818, Recall: 0.8435, F1 Score: 0.8028

# Fold 2, Epoch 10, Loss: 0.0035, Train Acc: 0.9988
# Fold 2, Epoch 20, Loss: 0.0008, Train Acc: 0.9999
# Fold 2, Epoch 30, Loss: 0.0052, Train Acc: 0.9986
# Fold 2, Epoch 40, Loss: 0.0002, Train Acc: 1.0000
# Fold 2, Epoch 50, Loss: 0.0002, Train Acc: 1.0000
# Fold 2, Epoch 60, Loss: 0.0002, Train Acc: 1.0000
# Fold 2, Epoch 70, Loss: 0.0027, Train Acc: 0.9992
# Fold 2, Epoch 80, Loss: 0.0019, Train Acc: 0.9994
# Fold 2, Epoch 90, Loss: 0.0003, Train Acc: 0.9999
# Fold 2, Epoch 100, Loss: 0.0001, Train Acc: 1.0000
# Fold 2 - Confusion Matrix:
# [[206   0   0   0   0   0   0   0   0   0]
#  [  0 203   0   0   0   0   0   0   6   0]
#  [  0   0 200   0   0   0   0   2   0   0]
#  [  0   0   0 200   0   0   0   0   0   0]
#  [  0   0   1  41 155   0   1  29   2   0]
#  [  0   0   0   0   0 203   0   0   0   0]
#  [  0   0  15   0   0   0 131  16   0   0]
#  [  0   0 134   0   0   0   2  48   0   0]
#  [  9   1   0   0   0   0   0   0 192   0]
#  [  0   0   0   0   0   0   0   0   0 203]]
# Fold 2 - Test Loss: 1.1563, Test Accuracy: 87.05%, Precision: 0.8835, Recall: 0.8705, F1 Score: 0.8627

# Fold 3, Epoch 10, Loss: 0.0170, Train Acc: 0.9948
# Fold 3, Epoch 20, Loss: 0.0025, Train Acc: 0.9988
# Fold 3, Epoch 30, Loss: 0.0073, Train Acc: 0.9978
# Fold 3, Epoch 40, Loss: 0.0060, Train Acc: 0.9981
# Fold 3, Epoch 50, Loss: 0.0001, Train Acc: 1.0000
# Fold 3, Epoch 60, Loss: 0.0023, Train Acc: 0.9995
# Fold 3, Epoch 70, Loss: 0.0001, Train Acc: 1.0000
# Fold 3, Epoch 80, Loss: 0.0016, Train Acc: 0.9995
# Fold 3, Epoch 90, Loss: 0.0002, Train Acc: 1.0000
# Fold 3, Epoch 100, Loss: 0.0000, Train Acc: 1.0000
# Fold 3 - Confusion Matrix:
# [[212   0   0   0   0   0   0   0   3   0]
#  [  0 181   0   0   0   0   0   0  18   0]
#  [  0   0 166   0   0   0   0  38   0   0]
#  [ 16   9   0 162   1   0   0   0   1   0]
#  [  0   1   1   0 195   0   0   0   0   0]
#  [  0   0   0   0   0 210   0   0   0   2]
#  [  0   0   0   0   0   0 217   0   0   0]
#  [  0   0   3   0   0   0   0 178   0   0]
#  [ 21   0   0   0   0   0   0   0 174   0]
#  [  0   0   1   0   0   0   0   0   0 190]]
# Fold 3 - Test Loss: 0.3277, Test Accuracy: 94.25%, Precision: 0.9475, Recall: 0.9425, F1 Score: 0.9426

# Fold 4, Epoch 10, Loss: 0.0104, Train Acc: 0.9971
# Fold 4, Epoch 20, Loss: 0.0020, Train Acc: 0.9995
# Fold 4, Epoch 30, Loss: 0.0001, Train Acc: 1.0000
# Fold 4, Epoch 40, Loss: 0.0025, Train Acc: 0.9992
# Fold 4, Epoch 50, Loss: 0.0019, Train Acc: 0.9994
# Fold 4, Epoch 60, Loss: 0.0068, Train Acc: 0.9984
# Fold 4, Epoch 70, Loss: 0.0127, Train Acc: 0.9975
# Fold 4, Epoch 80, Loss: 0.0025, Train Acc: 0.9992
# Fold 4, Epoch 90, Loss: 0.0001, Train Acc: 1.0000
# Fold 4, Epoch 100, Loss: 0.0002, Train Acc: 1.0000
# Fold 4 - Confusion Matrix:
# [[ 81   0   0   0   0   0   0   0  88   0]
#  [  0 185   0   0   0   0   0   0   7   0]
#  [  0   0 180   0   0   0   0  33   0   0]
#  [  0   0   0 213   3   0   0   0   0   0]
#  [  0   0   1   0 143   0  19   5   0   0]
#  [  0   0   0   0   0 199   0   0   0   0]
#  [  0   0   2   0   0   0 206   2   0   0]
#  [  0   0   1   0   0   0   0 211   0   0]
#  [  5   0   0   0   0   0   0   1 184   0]
#  [  0   0   0   0   0   0   0   0   0 231]]
# Fold 4 - Test Loss: 0.4825, Test Accuracy: 91.65%, Precision: 0.9326, Recall: 0.9165, F1 Score: 0.9136

# Fold 5, Epoch 10, Loss: 0.0078, Train Acc: 0.9979
# Fold 5, Epoch 20, Loss: 0.0257, Train Acc: 0.9926
# Fold 5, Epoch 30, Loss: 0.0005, Train Acc: 0.9999
# Fold 5, Epoch 40, Loss: 0.0006, Train Acc: 0.9998
# Fold 5, Epoch 50, Loss: 0.0013, Train Acc: 0.9998
# Fold 5, Epoch 60, Loss: 0.0017, Train Acc: 0.9998
# Fold 5, Epoch 70, Loss: 0.0000, Train Acc: 1.0000
# Fold 5, Epoch 80, Loss: 0.0000, Train Acc: 1.0000
# Fold 5, Epoch 90, Loss: 0.0004, Train Acc: 0.9999
# Fold 5, Epoch 100, Loss: 0.0005, Train Acc: 0.9998
# Fold 5 - Confusion Matrix:
# [[198   0   1   0   0   0   0   0   3   0]
#  [  0 158   0  40   0   0   0   0   0   0]
#  [  0   0 181   0   0   0   0   8   0   0]
#  [ 17  17   0  98  21   0   0   0  41   0]
#  [  0   0   0   0 194   0   0   1   2   0]
#  [  0   0   0   0   0 200   0   0   0   0]
#  [  0   0   0   0   0   0 200   0   0   0]
#  [  0   0   5   0   0   0   0 214   0   0]
#  [  3   0   0   0   0   0   0   0 198   0]
#  [  0   0   0   0   0   0   0   0   0 200]]
# Fold 5 - Test Loss: 0.4677, Test Accuracy: 92.05%, Precision: 0.9170, Recall: 0.9205, F1 Score: 0.9156

# Average Cross-Validation Loss: 0.7458
# Average Cross-Validation Accuracy: 89.87%
# Average Precision: 0.8925
# Average Recall: 0.8987
# Average F1 Score: 0.8874
