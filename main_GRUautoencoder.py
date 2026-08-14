import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from models.RangeFractionation import GaussianRangeFractionation
from models.GRU_Autoencoder import GRUAutoEncoder
from data_graphs.visualizer import Visualizer


"""Script utilizes range-fractionated data for GRU autoencoder to lower feature dimensions 
and K-Means clustering to cluster joint kinematics and GRF data to either stance or swing phase"""
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 0. Set Random Seed to Make Models Deterministic
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def deterministic(seed=42):
    # sets seeds (starting point) for python and pytorch random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True) # makes sure algorithms used in models are deterministic

deterministic(seed=42)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Load and Pre-process Features
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
raw_df = pd.read_csv("csv_trajectory_datasets/gait_phase_master_trajectories.csv")
main_df = raw_df.drop(columns=['time']) # only for training autoencoder, already indexed by timestep
LH_CTr_theta, LH_TrF_theta, LH_FTi_theta, LH_CTr_vel, LH_TrF_vel, LH_FTi_vel, effector_acc, GRF_z = Visualizer.extract_features(raw_df=raw_df) # extract features for plotting

LH_thetas = np.column_stack((LH_CTr_theta, LH_TrF_theta, LH_FTi_theta))
LH_vels = np.column_stack((LH_CTr_vel, LH_TrF_vel, LH_FTi_vel))
LH_eff_acc_z = effector_acc.to_numpy().reshape(-1, 1)

scaler = StandardScaler()
scaled_LH_thetas = scaler.fit_transform(LH_thetas)
scaled_LH_vels = scaler.fit_transform(LH_vels)
scaled_lin_acc_z = scaler.fit_transform(LH_eff_acc_z)

# sequence function adaptation for autoencoder input
def create_seq(data, seq_length):
    X= []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i+seq_length]) #past values
    return np.array(X)

SEQ_LENGTH = 6 
X_theta = create_seq(scaled_LH_thetas, seq_length=SEQ_LENGTH) 
X_vel = create_seq(scaled_LH_vels, seq_length=SEQ_LENGTH) 
X_eff_acc_z = create_seq(scaled_lin_acc_z, seq_length=SEQ_LENGTH)

# convert to tensor
X_theta_tensor = torch.from_numpy(X_theta).float()
X_vel_tensor = torch.from_numpy(X_vel).float()
X_eff_acc_z_tensor = torch.from_numpy(X_eff_acc_z).float()

print(f"Input shape (joint angles): {X_theta_tensor.shape}")
print(f"Input shape (velocity): {X_vel_tensor.shape}")

# get number of features for each sesnor stream to be used in model initialization
orig_num_feature_theta = X_theta_tensor.shape[2] 
orig_num_feature_vel = X_vel_tensor.shape[2] 
orig_num_feature_eff_acc_z = X_eff_acc_z_tensor.shape[2]

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Initiazlize Models
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model = GRUAutoEncoder(theta_num_features=orig_num_feature_theta, vel_num_features=orig_num_feature_vel, lin_acc_num_features=orig_num_feature_eff_acc_z,
                        theta_neurons=1600, vel_neurons=3200, lin_acc_neurons=1600,
                           hidden_dim=64, latent_dim=1) # GRU autoencoder model -> want latent_dim to be 1 to convert to sigmoid

# loss and optimizer definitions
criterion = nn.MSELoss() # mean squared error loss for data reconstruction
optimizer= torch.optim.Adam(model.parameters(), lr = 0.001)

#define loss weights (velocity more important than joint angles for swing phase detection)
BETA = 1.0   #weight for joint angles
GAMMA = 3.0  #weight for velocity
SIGMA = 1.0 #weight for vertical linear acceleration of effector

# concatenate into a tensor dataset and then dataloader to go through each tensor in chunks of 64
dataset = TensorDataset(X_theta_tensor, X_vel_tensor, X_eff_acc_z_tensor)
dataloader = DataLoader(dataset, batch_size=64)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Training Loop 
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# focus is on optimizing reconstruction loss to gain more representative latent embedding
epochs = 100
print("Training model... ")
for epoch in range(epochs):
    model.train()
    
    epoch_loss = 0.0
    
    for batch_theta, batch_vel, batch_eff_acc_z in dataloader:
        optimizer.zero_grad() 
        
        #forward pass on the mini-batch
        recon_theta, recon_vel, recon_eff_acc_z, _ = model(batch_theta, batch_vel, batch_eff_acc_z) 
        
        # compute individual losses
        loss_theta = criterion(recon_theta, batch_theta) 
        loss_vel = criterion(recon_vel, batch_vel) 
        loss_eff_acc_z = criterion(recon_eff_acc_z, batch_eff_acc_z)
        
        # weighted loss combination
        loss = (BETA * loss_theta) + (GAMMA * loss_vel) + (SIGMA * loss_eff_acc_z)
        
        loss.backward() # backprop
        optimizer.step()  # update parameter

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training complete.")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. Testing
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model.eval() # stop training
recons = []
with torch.no_grad():
    for _ in range(30):  # run multiple times to get different dropout samples
        recon_theta, recon_vel, recon_eff_acc_z, probs = model(X_theta_tensor, X_vel_tensor, X_eff_acc_z_tensor) # get probabilities and reconstructed inputs
        recons.append(probs) # save probabilities of each run. dropout will cause some variation in the probabilities, which can be used to estimate uncertainty.

# take mean of the reconstructed probabilities
avg_probs = torch.stack(recons).mean(dim=0)

# calculate variance of the reconstructed probabilities for uncertainty estimation
prob_variance = torch.stack(recons).var(dim=0)
uncertainty_score = prob_variance.mean(dim=-1)  # average variance across all samples as a single uncertainty score
print(f"Uncertainty score (average variance across all samples): {uncertainty_score.max().item():.4f}") # max uncertainty score across all samples

swing_phase_detection = (avg_probs < 0.5).int()  # convert probabilities to binary labels. swing phase is 0, stance phase is 1
swing_phase_indices = np.where(swing_phase_detection == 0)[0] 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. Plot Data
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# plot raw features before range fractionation
Visualizer.plot_features(raw_df["time"], LH_CTr_theta, LH_TrF_theta, LH_FTi_theta, LH_CTr_vel, LH_TrF_vel, LH_FTi_vel, effector_acc)

# plot range fractionated input for thetas as example of what it looks like
range_frac_inst = GaussianRangeFractionation(num_neurons=1, min_value=-1.0, max_value=1.0)
theta_frac = range_frac_inst(X_theta_tensor).detach().cpu().numpy()

Visualizer.plot_rangeFrac_example(theta_frac=theta_frac)

# GRFs over time in cartesian coords with labels
Visualizer.plot_labeledGRF(raw_df=raw_df, raw_data_GRF=GRF_z, SEQ_LENGTH=SEQ_LENGTH, swing_indices=swing_phase_indices)