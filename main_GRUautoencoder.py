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
    torch.use_deterministic_algorithms(True) # makes sure algorithms used in models are deterministic

deterministic(seed=42)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Load and Pre-process Features
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
raw_df = pd.read_csv("csv_trajectory_datasets/gait_phase_LH_trajectories.csv")
main_df = raw_df.drop(columns=['time']) # only for training autoencoder, already indexed by timestep
n_states = 2 # for clustering stance and swing phases

# preprocess data
def scale_dataframe(dataframe): # seperates dataframe into different sensor streams and z-norms them
    scaler = StandardScaler()
    raw_data = dataframe.values.astype(np.float32)

    raw_data_GRF = raw_data[:, :3]  # extract ground reaction force (GRF) data

    raw_data_theta = raw_data[:, [3,5,7]]  # extract joint angle data
    scaled_data_theta = scaler.fit_transform(raw_data_theta)

    raw_data_vel = raw_data[:, [4,6,8]]  #extract joint velocity data
    scaled_data_vel = scaler.fit_transform(raw_data_vel)
    return scaled_data_theta, scaled_data_vel, raw_data_GRF, raw_data_theta, raw_data_vel

# sequence function adaptation for autoencoder input
def create_seq(data, seq_length):
    X= []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i+seq_length]) #past values
    return np.array(X)

scaled_theta, scaled_vel, raw_data_GRF, raw_data_theta, raw_data_vel = scale_dataframe(main_df) # raw data to be used for plotting

SEQ_LENGTH = 6 
X_theta = create_seq(scaled_theta, seq_length=SEQ_LENGTH) 
X_vel = create_seq(scaled_vel, seq_length=SEQ_LENGTH) 

# convert to tensor
X_theta_tensor = torch.from_numpy(X_theta).float()
X_vel_tensor = torch.from_numpy(X_vel).float()

print(f"Input shape (joint angles): {X_theta_tensor.shape}")
print(f"Input shape (velocity): {X_vel_tensor.shape}")

# get number of features for each sesnor stream to be used in model initialization
orig_num_feature_theta = X_theta_tensor.shape[2] 
orig_num_feature_vel = X_vel_tensor.shape[2] 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Initiazlize Models
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model = GRUAutoEncoder(theta_num_features=orig_num_feature_theta, vel_num_features=orig_num_feature_vel,
                        theta_neurons=1600, vel_neurons=1600,
                           hidden_dim=64, latent_dim=1) # GRU autoencoder model -> want latent_dim to be 1 to convert to sigmoid

# loss and optimizer definitions
criterion = nn.MSELoss() # mean squared error loss for data reconstruction
optimizer= torch.optim.Adam(model.parameters(), lr = 0.001)

#define loss weights (velocity more important than joint angles for swing phase detection)
BETA = 1.0   #weight for joint angles
GAMMA = 3.0  #weight for velocity

# concatenate into a tensor dataset and then dataloader to go through each tensor in chunks of 64
dataset = TensorDataset(X_theta_tensor, X_vel_tensor)
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
    
    for batch_theta, batch_vel in dataloader:
        optimizer.zero_grad() 
        
        #forward pass on the mini-batch
        recon_theta, recon_vel, _ = model(batch_theta, batch_vel) 
        
        # compute individual losses
        loss_theta = criterion(recon_theta, batch_theta) 
        loss_vel = criterion(recon_vel, batch_vel) 
        
        # weighted loss combination
        loss = (BETA * loss_theta) + (GAMMA * loss_vel)
        
        loss.backward() # backprop
        optimizer.step()  # update parameter

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training complete.")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. Testing
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model.eval() # stop training
with torch.no_grad():
    recon_theta, recon_vel, probs = model(X_theta_tensor, X_vel_tensor) # get probabilities
    swing_phase_detection = (probs >= 0.5).int()  # convert probabilities to binary labels
swing_phase_indices = np.where(swing_phase_detection == 0)[0]

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. Plot Data
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
LH_CTr_theta, LH_TrF_theta, LH_FTi_theta, LH_CTr_vel, LH_TrF_vel, LH_FTi_vel, GRF_x, GRF_y, GRF_z = Visualizer.extract_features(raw_df=raw_df) 

# plot raw features before range fractionation
Visualizer.plot_features(raw_df["time"], LH_CTr_theta, LH_TrF_theta, LH_FTi_theta, LH_CTr_vel, LH_TrF_vel, LH_FTi_vel)

# plot range fractionated input for thetas as example of what it looks like
range_frac_inst = GaussianRangeFractionation(num_neurons=1, min_value=-1.0, max_value=1.0)
theta_frac = range_frac_inst(X_theta_tensor).detach().cpu().numpy()

Visualizer.plot_rangeFrac_example(theta_frac=theta_frac)

# angles over time of each joint with labels
#Visualizer.plot_labeledTheta(raw_df=raw_df, raw_data_theta=raw_data_theta, SEQ_LENGTH=SEQ_LENGTH, swing_indices=swing_phase_indices)

# GRFs over time in cartesian coords with labels
Visualizer.plot_labeledGRF(raw_df=raw_df, raw_data_GRF=raw_data_GRF, SEQ_LENGTH=SEQ_LENGTH, swing_indices=swing_phase_indices)