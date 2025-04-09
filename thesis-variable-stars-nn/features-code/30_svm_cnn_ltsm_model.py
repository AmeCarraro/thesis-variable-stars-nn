import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

# Define file paths for data and log output
data_path = "C:\\Users\\carra\\Unipd\\tesi\\data\\"
log_file_path = "C:\\Users\\carra\\Unipd\\tesi\\results\\feature_selection_results_30svm.txt"

# List of all feature names and their total count
feature_names = [
    "Multiband_period",
    "Period_band_r",
    "Harmonics_mag_2_r",
    "Harmonics_phase_2_r",
    "MHPS_low_r",
    "IAR_phi_r",
    "GP_DRW_tau_r",
    "Harmonics_mag_1_r",
    "iqr_r",
    "MHPS_high_r",
    "Skew_r",
    "Harmonics_phase_3_r",
    "Power_rate_1/3",
    "MHPS_non_zero_r",
    "Mean_r",
    "Psi_eta_r",
    "Psi_CS_r",
    "Harmonics_mse_r",
    "PPE",
    "Eta_e_r",
    "StetsonK_r",
    "Harmonics_mag_7_r",
    "SmallKurtosis_r",
    "Power_rate_4",
    "delta_period_r",
    "Beyond1Std_r",
    "MaxSlope_r",
    "Harmonics_phase_6_r",
    "LinearTrend_r",
    "Pvar_r"
]
num_features = len(feature_names)

# Define the CNN-LSTM model architecture
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=2, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, padding=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4)  # 4 output classes

    def forward(self, x):
        # Reshape input to (batch_size, channels, feature_length)
        x = x.view(x.size(0), 1, x.size(1))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.maxpool2(x)
        # Rearrange dimensions for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Extract the output from the final time step
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load train and test datasets from text files
X_train = np.loadtxt(f"{data_path}featureTR_30_svm.txt")
y_train = np.loadtxt(f"{data_path}labelTR_30_svm.txt")
X_test = np.loadtxt(f"{data_path}featureTE_30_svm.txt")
y_test = np.loadtxt(f"{data_path}labelTE_30_svm.txt")

# Training configuration parameters
num_epochs = 100
batch_size = 32
learning_rate = 0.0002

# Open log file for writing results
with open(log_file_path, 'w') as log_file:
    best_config = None
    best_accuracy = 0.0
    best_f1 = 0.0
    best_num_features = 0

    # Loop over increasing number of features (from 1 to total number)
    for k in range(1, num_features + 1):
        # Select the first k features
        feature_indices = list(range(k))
        X_train_sub = X_train[:, feature_indices]
        X_test_sub = X_test[:, feature_indices]

        # Convert numpy arrays to torch tensors
        X_train_tensor = torch.tensor(X_train_sub, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_test_tensor = torch.tensor(X_test_sub, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        # Create dataloaders for training and testing
        dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataset_test = TensorDataset(X_test_tensor, y_test_tensor)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        # Initialize model, loss function and optimizer
        model = CNN_LSTM().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in dataloader_train:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluation on the test dataset
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
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        cm = confusion_matrix(all_labels, all_predictions)

        # Prepare log message for the current configuration
        used_feature_names = [feature_names[i] for i in feature_indices]
        comb_str = ", ".join(used_feature_names)
        log_message = (
            f"Using {k} features: {comb_str}\n"
            f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, F1 Score: {f1}\n"
            f"Confusion Matrix:\n{cm}\n"
        )
        log_file.write(log_message)
        print(log_message.strip())

        # Update best configuration if current F1 score is higher
        if f1 > best_f1:
            best_f1 = f1
            best_accuracy = test_accuracy
            best_config = used_feature_names
            best_num_features = k

    # Log and print a summary of the best configuration
    summary = (
        f"\nBest configuration: using {best_num_features} features ({', '.join(best_config)}) "
        f"with Accuracy: {best_accuracy}, F1: {best_f1}\n"
    )
    log_file.write(summary)
    print(summary.strip())





#Using 1 features: Multiband_period
#Test Loss: 0.13243562932170572, Test Accuracy: 0.9765, F1 Score: 0.9764969027522874
#Using 2 features: Multiband_period, Period_band_r
#Test Loss: 0.212543001132352, Test Accuracy: 0.9515, F1 Score: 0.9516391039828114
#Using 3 features: Multiband_period, Period_band_r, Harmonics_mag_2_r
#Test Loss: 0.31430301334827193, Test Accuracy: 0.945, F1 Score: 0.9450418681033591
#Using 4 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r
#Test Loss: 0.2375115412287414, Test Accuracy: 0.931, F1 Score: 0.9310119642121881
#Using 5 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r
#Test Loss: 0.234707384133741, Test Accuracy: 0.9505, F1 Score: 0.9506888311684809
#Using 6 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r
#Test Loss: 0.38340446615355117, Test Accuracy: 0.924, F1 Score: 0.9247507461167017
#Using 7 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r
#Test Loss: 0.2868624654051567, Test Accuracy: 0.932, F1 Score: 0.9324024882178764
#Using 8 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r
#Test Loss: 0.30238638166338205, Test Accuracy: 0.923, F1 Score: 0.9237693358789387
#Using 9 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r
#Test Loss: 0.4700409345151413, Test Accuracy: 0.918, F1 Score: 0.9181039580337568
#Using 10 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r
#Test Loss: 0.2775624044653442, Test Accuracy: 0.9185, F1 Score: 0.9186686688191779
#Using 11 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r
#Test Loss: 0.22975919433381586, Test Accuracy: 0.9525, F1 Score: 0.9525036404541162
#Using 12 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r
#Test Loss: 0.2601815416618058, Test Accuracy: 0.94, F1 Score: 0.9403335357343315
#Using 13 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3
#Test Loss: 0.2449366722444427, Test Accuracy: 0.9425, F1 Score: 0.9425308884135447
#Using 14 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r
#Test Loss: 0.1767994744762305, Test Accuracy: 0.9545, F1 Score: 0.9545685515875895
#Using 15 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r
#Test Loss: 0.20086610102139058, Test Accuracy: 0.9485, F1 Score: 0.9484771153795605
#Using 16 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r
#Test Loss: 0.23311853179678557, Test Accuracy: 0.944, F1 Score: 0.9441642048426298
#Using 17 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r
#Test Loss: 0.21374856693566674, Test Accuracy: 0.9495, F1 Score: 0.9494890091785911
#Using 18 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r
#Test Loss: 0.1682162921772235, Test Accuracy: 0.957, F1 Score: 0.9572044503339836
#Using 19 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE
#Test Loss: 0.1967167864786461, Test Accuracy: 0.953, F1 Score: 0.9530366342535206
#Using 20 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r
#Test Loss: 0.26041749414677423, Test Accuracy: 0.9435, F1 Score: 0.9436118121244207
#Using 21 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r, StetsonK_r
#Test Loss: 0.18310556740366987, Test Accuracy: 0.9495, F1 Score: 0.9496783884163922
#Using 22 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r, StetsonK_r, Harmonics_mag_7_r
#Test Loss: 0.21297784547306717, Test Accuracy: 0.9385, F1 Score: 0.9385489311711921
#Using 23 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r, StetsonK_r, Harmonics_mag_7_r, SmallKurtosis_r    
#Test Loss: 0.2217710503406586, Test Accuracy: 0.9505, F1 Score: 0.9506873638095719
#Using 24 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r, StetsonK_r, Harmonics_mag_7_r, SmallKurtosis_r, Power_rate_4
#Test Loss: 0.2965562815467517, Test Accuracy: 0.908, F1 Score: 0.9080715672409042
#Using 25 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r, StetsonK_r, Harmonics_mag_7_r, SmallKurtosis_r, Power_rate_4, delta_period_r
#Test Loss: 0.23998179447851956, Test Accuracy: 0.9345, F1 Score: 0.9351186256157394
#Using 26 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r, StetsonK_r, Harmonics_mag_7_r, SmallKurtosis_r, Power_rate_4, delta_period_r, Beyond1Std_r
#Test Loss: 0.216311413968455, Test Accuracy: 0.9405, F1 Score: 0.9404332960873387
#Using 27 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r, StetsonK_r, Harmonics_mag_7_r, SmallKurtosis_r, Power_rate_4, delta_period_r, Beyond1Std_r, MaxSlope_r
#Test Loss: 0.2113361255938394, Test Accuracy: 0.942, F1 Score: 0.9422048917735992
#Using 28 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r, StetsonK_r, Harmonics_mag_7_r, SmallKurtosis_r, Power_rate_4, delta_period_r, Beyond1Std_r, MaxSlope_r, Harmonics_phase_6_r
#Test Loss: 0.21399298058601007, Test Accuracy: 0.9515, F1 Score: 0.95157691208813
#Using 29 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r, StetsonK_r, Harmonics_mag_7_r, SmallKurtosis_r, Power_rate_4, delta_period_r, Beyond1Std_r, MaxSlope_r, Harmonics_phase_6_r, LinearTrend_r
#Test Loss: 0.22559553490848178, Test Accuracy: 0.9385, F1 Score: 0.9386914545305892
#Using 30 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r, Harmonics_mag_1_r, iqr_r, MHPS_high_r, Skew_r, Harmonics_phase_3_r, Power_rate_1/3, MHPS_non_zero_r, Mean_r, Psi_eta_r, Psi_CS_r, Harmonics_mse_r, PPE, Eta_e_r, StetsonK_r, Harmonics_mag_7_r, SmallKurtosis_r, Power_rate_4, delta_period_r, Beyond1Std_r, MaxSlope_r, Harmonics_phase_6_r, LinearTrend_r, Pvar_r
#Test Loss: 0.14360010743303786, Test Accuracy: 0.9665, F1 Score: 0.966487365596794
#Best configuration: using 1 features (Multiband_period) with Accuracy: 0.9765, F1: 0.9764969027522874


# Using 1 features: Multiband_period
# Test Loss: 0.13386235847359612, Test Accuracy: 0.9755, F1 Score: 0.9754598040216587
# Confusion Matrix:
# [[492   5   3   0]
#  [ 13 474  13   0]
#  [  2   5 489   4]
#  [  2   0   2 496]]

# Using 2 features: Multiband_period, Period_band_r
# Test Loss: 0.20851944132693231, Test Accuracy: 0.965, F1 Score: 0.9651541813281267
# Confusion Matrix:
# [[473  25   2   0]
#  [  6 485   9   0]
#  [  2  10 487   1]
#  [  1   3  11 485]]

# Using 3 features: Multiband_period, Period_band_r, Harmonics_mag_2_r
# Test Loss: 0.21701667192465965, Test Accuracy: 0.966, F1 Score: 0.9659586839655635
# Confusion Matrix:
# [[493   3   3   1]
#  [ 23 467   9   1]
#  [  2   8 487   3]
#  [  1   9   5 485]]

# Using 4 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r
# Test Loss: 0.3507131679308793, Test Accuracy: 0.9205, F1 Score: 0.9203947203863839
# Confusion Matrix:
# [[395 102   3   0]
#  [ 14 473  13   0]
#  [  5   9 485   1]
#  [  1   1  10 488]]

# Using 5 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r
# Test Loss: 0.19408205280169136, Test Accuracy: 0.949, F1 Score: 0.9491968676175047
# Confusion Matrix:
# [[470  27   3   0]
#  [ 24 465  11   0]
#  [  4  15 478   3]
#  [  1   6   8 485]]

# Using 6 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r
# Test Loss: 0.19971853538253714, Test Accuracy: 0.948, F1 Score: 0.9479970094167166
# Confusion Matrix:
# [[471  25   4   0]
#  [ 26 452  22   0]
#  [  2   9 486   3]
#  [  1   8   4 487]]

# Using 7 features: Multiband_period, Period_band_r, Harmonics_mag_2_r, Harmonics_phase_2_r, MHPS_low_r, IAR_phi_r, GP_DRW_tau_r
# Test Loss: 0.23262002080723287, Test Accuracy: 0.9425, F1 Score: 0.942795719287355
# Confusion Matrix:
# [[464  32   3   1]
#  [ 30 460  10   0]
#  [  1  18 479   2]
#  [  1   8   9 482]]

# Using 8 features: Multiband_period, ..., Harmonics_mag_1_r
# Test Loss: 0.40072027949379785, Test Accuracy: 0.9105, F1 Score: 0.9108363746070675
# Confusion Matrix:
# [[396  99   4   1]
#  [ 31 458  11   0]
#  [  4  14 480   2]
#  [  1   7   5 487]]

# ...

# (Tutti i blocchi continuano con lo stesso schema)

# Using 30 features: Multiband_period, ..., Pvar_r
# Test Loss: 0.25956543033853885, Test Accuracy: 0.928, F1 Score: 0.928319369306293
# Confusion Matrix:
# [[478  16   5   1]
#  [ 36 456   8   0]
#  [  9  45 446   0]
#  [  2   3  19 476]]

# Best configuration: using 1 features (Multiband_period) with Accuracy: 0.9755, F1: 0.9754598040216587
