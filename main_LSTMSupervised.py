import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from models.RangeFractionation import GaussianRangeFractionation
from models.LSTMmodel import LSTMRegressionModel


"""Script trains and evaluates a supervised LSTM model for ground reaction force predictions in the cartesian space of the left front leg."""
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Load and preprocess the data
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
num_translations = 7
num_rotations = 11
seq_length = 50

all_train_sequences = []
all_train_targets = []
all_test_sequences = []
all_test_targets = []

# intialize scalers for features and targets
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# load and fit scalers on training data split
print("Fitting scalers on training data split...")
train_trajectories = []
test_trajectories = []

# generate a list of all file combinations and split them (e.g., 80% of trajectories for training)
all_combinations = [(t, r) for t in range(num_translations) for r in range(num_rotations)]
np.random.seed(42) # For reproducibility
torch.manual_seed(42) # For reproducibility
torch.use_deterministic_algorithms(True) # For reproducibility
#np.random.shuffle(all_combinations)

split_idx = int(0.8 * len(all_combinations))
train_combos = all_combinations[:split_idx]
test_combos = all_combinations[split_idx:]

# fit features and targets sequentially using chunks
for tran_idx, rot_idx in train_combos:
    df = pd.read_csv(f"csv_trajectory_datasets/master_trajectories_tran{tran_idx}_rot{rot_idx}.csv")
    
    feats = df[["LF_TrF_angle", "LF_FTi_angle", "LF_TrF_vel", "LF_FTi_vel"]].values
    targs = df[["LF_Tip_GRF_x", "LF_Tip_GRF_y", "LF_Tip_GRF_z"]].values
    
    # add noise to features before fitting scaler
    noise = np.random.normal(0, 0.01, feats.shape)
    feats += noise
    
    feature_scaler.partial_fit(feats)
    target_scaler.partial_fit(targs)

# process individual trajectories
def process_trajectory(tran_idx, rot_idx, feat_scaler, targ_scaler, is_train=True):
    df = pd.read_csv(f"csv_trajectory_datasets/master_trajectories_tran{tran_idx}_rot{rot_idx}.csv")
    
    feats = df[["LF_TrF_angle", "LF_FTi_angle", "LF_TrF_vel", "LF_FTi_vel"]].values
    targs = df[["LF_Tip_GRF_x", "LF_Tip_GRF_y", "LF_Tip_GRF_z"]].values
    
    # transform using global scalers
    feats_scaled = feat_scaler.transform(feats)
    targs_scaled = targ_scaler.transform(targs)
    
    # create windows ONLY within this specific trajectory file
    sequences, target_sequences = [], []
    for i in range(len(feats_scaled) - seq_length):
        sequences.append(feats_scaled[i:i + seq_length])
        target_sequences.append(targs_scaled[i + seq_length]) # Next-step prediction
        
    return sequences, target_sequences

# process Train Files
print("Processing training sequences...")
for tran_idx, rot_idx in train_combos:
    seqs, targs = process_trajectory(tran_idx, rot_idx, feature_scaler, target_scaler, is_train=True)
    all_train_sequences.extend(seqs)
    all_train_targets.extend(targs)

# process Test Files
print("Processing testing sequences...")
for tran_idx, rot_idx in test_combos:
    seqs, targs = process_trajectory(tran_idx, rot_idx, feature_scaler, target_scaler, is_train=False)
    all_test_sequences.extend(seqs)
    all_test_targets.extend(targs)

# 6. Convert straight to PyTorch Tensors
train_sequences_tensor = torch.tensor(np.array(all_train_sequences), dtype=torch.float32)
train_target_sequences_tensor = torch.tensor(np.array(all_train_targets), dtype=torch.float32)

test_sequences_tensor = torch.tensor(np.array(all_test_sequences), dtype=torch.float32)
test_target_sequences_tensor = torch.tensor(np.array(all_test_targets), dtype=torch.float32)

print(f"Final Train Shapes: {train_sequences_tensor.shape}, Targets: {train_target_sequences_tensor.shape}")
print(f"Final Test Shapes: {test_sequences_tensor.shape}, Targets: {test_target_sequences_tensor.shape}")

train_dataset = TensorDataset(train_sequences_tensor, train_target_sequences_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

test_dataset = TensorDataset(test_sequences_tensor, test_target_sequences_tensor)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Define LSTM Model, optimizer parameters, criterion and start training loop
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model = LSTMRegressionModel(input_size=train_sequences_tensor.shape[2], hidden_size=64, num_layers=3, output_size=train_target_sequences_tensor.shape[1])  # 4 input features, 3 output targets
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Starting training loop...")
for epoch in range(50):  # number of epochs
    model.train()
    running_loss = 0.0

    for batch_seqs, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_seqs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() 

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/50], AverageLoss: {running_loss / len(train_loader):.4f}') # print average loss for the epoch

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Evaluate the model on the test set
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def generate_predictions(model, tran_idx, rot_idx, feat_scaler, targ_scaler):
    model.eval()
    with torch.no_grad():
        df = pd.read_csv(f"csv_trajectory_datasets/master_trajectories_tran{tran_idx}_rot{rot_idx}.csv")
        feats = df[["LF_TrF_angle", "LF_FTi_angle", "LF_TrF_vel", "LF_FTi_vel"]].values
        targs = df[["LF_Tip_GRF_x", "LF_Tip_GRF_y", "LF_Tip_GRF_z"]].values
        
        feats_scaled = feat_scaler.transform(feats)
        targs_scaled = targ_scaler.transform(targs)
        
        sequences, target_sequences = [], []
        for i in range(len(feats_scaled) - seq_length):
            sequences.append(feats_scaled[i:i + seq_length])
            target_sequences.append(targs_scaled[i + seq_length]) # Next-step prediction
        
        sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
        predictions_scaled = model(sequences_tensor).numpy()
        
        # Inverse transform to get back to original scale
        predictions = targ_scaler.inverse_transform(predictions_scaled)
        
    return predictions, targs[seq_length:]  # Return predictions and corresponding true targets

predictions, true_targets = generate_predictions(model, tran_idx=6, rot_idx=10, feat_scaler=feature_scaler, targ_scaler=target_scaler) # Evaluate on a specific trajectory (tran6_rot10)

figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,10), sharex=True)
ax1, ax2, ax3 = axes.flatten()
ax1.plot(predictions[:, 0], label="Predicted LF_Tip_GRF_x", color='blue')
ax1.plot(true_targets[:, 0], label="True LF_Tip_GRF_x", color='orange')
ax1.set_ylabel('Newton (N)', fontsize=5)
ax1.set_title('LF_Tip_GRF_x Predictions vs True Values', fontsize=10)

ax2.plot(predictions[:, 1], label="Predicted LF_Tip_GRF_y", color='blue')
ax2.plot(true_targets[:, 1], label="True LF_Tip_GRF_y", color='orange')
ax2.set_ylabel('Newton (N)', fontsize=5)
ax2.set_title('LF_Tip_GRF_y Predictions vs True Values', fontsize=10)

ax3.plot(predictions[:, 2], label="Predicted LF_Tip_GRF_z", color='blue')
ax3.plot(true_targets[:, 2], label="True LF_Tip_GRF_z", color='orange')
ax3.set_ylabel('Newton (N)', fontsize=5)
ax3.set_xlabel('Time Steps')
ax3.set_title('LF_Tip_GRF_z Predictions vs True Values', fontsize=10)
figure.suptitle('LSTM Model Predictions vs True Ground Reaction Forces')
plt.legend()
plt.show()