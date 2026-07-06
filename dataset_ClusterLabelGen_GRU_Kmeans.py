from xml.parsers.expat import model

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


"""Script utilizes a range-fractionated GRU autoencoder to lower feature dimensions 
and K-Means clustering to cluster joint kinematic and GRF data to either stance or swing phase"""

#initialize model
class GaussianRangeFractionatation(nn.Module): #to be used before autoencoder to fractionate the input data into a range of neurons (acts as encoder)
    def __init__(self, num_neurons, min_value=0.0, max_value=1.0, sigma=0.1):
        super(GaussianRangeFractionatation, self).__init__()
        self.num_neurons = num_neurons
        self.min_value = min_value
        self.max_value = max_value
        self.sigma = sigma
        self.centers = torch.linspace(min_value, max_value, num_neurons)  #centers of the Gaussian functions
        self.sigma = torch.full((num_neurons,), sigma)  #standard deviation for each Gaussian function

        thresholds = torch.linspace(min_value, max_value, num_neurons)
        self.register_buffer('thresholds', thresholds)  #register thresholds as a buffer (fixed tensor that is not a parameter)

    def forward(self, x):
        # expected input shape: [batch_size, seq_len, num_features]
        batch_size, seq_len, num_features = x.shape
        
        # expand input dimensions for broadcasting: [batch, seq_len, features, 1]
        x_expanded = x.unsqueeze(-1)
        
        # reshape centers & sigmas for broadcasting: [1, 1, 1, num_neurons]
        mu = self.centers.view(1, 1, 1, -1)
        sigma = self.sigma.view(1, 1, 1, -1)
        
        # compute Gaussian RBF Activation
        squared_diff = (x_expanded - mu) ** 2
        variance = 2.0 * (sigma ** 2) + 1e-8 # 1e-8 prevents division by zero if sigma -> 0
        activations = torch.exp(-squared_diff / variance)
        
        # [batch_size, seq_len, num_features * num_neurons]
        return activations.view(batch_size, seq_len, -1)

class GRUAutoEncoder(nn.Module):
    def __init__(self, GRF_num_features, pos_num_features, acc_num_features, GRF_neurons, pos_neurons, acc_neurons, hidden_dim, latent_dim):
        super(GRUAutoEncoder, self).__init__()
        self.GRF_num_features = GRF_num_features
        self.pos_num_features = pos_num_features
        self.acc_num_features = acc_num_features
        self.GRF_neurons = GRF_neurons
        self.pos_neurons = pos_neurons
        self.acc_neurons = acc_neurons
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # encoder expects (batch, seq_len, features)
        #seperate the input data into a range of neurons (populations) with seperate streams for each feature, then concatenate the streams into a single input for the GRU
        self.GRF_fractionation = GaussianRangeFractionatation(num_neurons=GRF_neurons, min_value=-1.0, max_value=1.0)  
        self.pos_fractionation = GaussianRangeFractionatation(num_neurons=pos_neurons, min_value=-1.0, max_value=1.0)  
        self.acc_fractionation = GaussianRangeFractionatation(num_neurons=acc_neurons, min_value=-1.0, max_value=1.0)  
        
        fractionated_GRF_input_dim = self.GRF_neurons * self.GRF_num_features 
        fractionated_pos_input_dim = self.pos_neurons * self.pos_num_features  
        fractionated_acc_input_dim = self.acc_neurons * self.acc_num_features 
        total_fractionated_input_dim = fractionated_GRF_input_dim + fractionated_pos_input_dim + fractionated_acc_input_dim

        
        """for encoder function, we will use a GRU to encode the fractionated input data into a latent space of dimension embed_dim"""
        #shared encoder
        self.encoder_gru = nn.GRU(input_size=total_fractionated_input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2)

        #latent embedding layer to reduce the hidden state to a lower dimension
        self.latent_layer = nn.Linear(hidden_dim, latent_dim) 
        
        #map latent embedding back to hidden dimension for decoder
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)

        #shared decoder
        self.decoder_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1)

        #output layer to map back to original feature space
        self.output2total = nn.Linear(hidden_dim, total_fractionated_input_dim)

        self.outputGRF = nn.Linear(total_fractionated_input_dim, self.GRF_num_features)  #output layer for GRF
        self.outputPos = nn.Linear(total_fractionated_input_dim, self.pos_num_features)  #output layer for position
        self.outputAcc = nn.Linear(total_fractionated_input_dim, self.acc_num_features)  #output layer for acceleration

    def forward(self, GRF_x, pos_x, acc_x):
        #fractionate input data
        #seq should be same value for all inputs, but batch size can be different
        GRF_batch_size, GRF_seq_len, _ = GRF_x.shape
        GRFfractionated_x = self.GRF_fractionation(GRF_x)  

        pos_batch_size, pos_seq_len, _ = pos_x.shape
        posfractionated_x = self.pos_fractionation(pos_x)  

        acc_batch_size, acc_seq_len, _ = acc_x.shape
        accfractionated_x = self.acc_fractionation(acc_x)  

        #concatenate fractionated inputs
        total_fractionated_x = torch.cat((GRFfractionated_x, posfractionated_x, accfractionated_x), dim=2)  #concatenate along feature dimension

        #encoder
        _, hidden = self.encoder_gru(total_fractionated_x)  #hidden shape: (num_layers, batch, hidden_dim)
        hidden = hidden[-1]  #take the last layer's hidden state (shape: (batch, hidden_dim))
        latent_embed = self.latent_layer(hidden)  #project to latent space
        latent_embed_expanded = latent_embed.unsqueeze(1).repeat(1, 1, 1)  #expand latent embedding to match sequence length
        #decoder
        latent_to_hidden = self.latent2hidden(latent_embed_expanded)  #map latent embedding back to hidden dimension
        decoder_output, _ = self.decoder_gru(latent_to_hidden)  #decoder output
        output_total = self.output2total(decoder_output)  #map back to total fractionated input dimension

        #seperate the outputs for GRF, position, and acceleration
        GRF_recon = self.outputGRF(output_total)  #reconstructed GRF
        pos_recon = self.outputPos(output_total)  #reconstructed position
        acc_recon = self.outputAcc(output_total)  #reconstructed acceleration

        return GRF_recon, pos_recon, acc_recon, latent_embed

n_states = 2 #for clustering stance and swing phases

#load dataset
raw_df = pd.read_csv("csv/gait_phase_LH_trajectories.csv").drop(columns=['time']) #only for training autoencoder, already indexed by timestep
time_vec = pd.read_csv("csv/gait_phase_LH_trajectories.csv")[['time']].values #for plotting data

#preprocess data
scaler = StandardScaler()
raw_data = raw_df.values.astype(np.float32)

raw_data_GRF = raw_data[:, :3]  #extract GRF data
scaled_data_GRF = scaler.fit_transform(raw_data_GRF)
GRF_dataLoader = torch.utils.data

raw_data_pos = raw_data[:, [3,5,7]]  #extract joint kinematics data
scaled_data_pos = scaler.fit_transform(raw_data_pos)

raw_data_acc = raw_data[:, [4,6,8]]  #extract joint kinematics data
scaled_data_acc = scaler.fit_transform(raw_data_acc)

#sequence function adaptation for autoencoder input
def create_seq(data, seq_length):
    X= []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i+seq_length]) #past values
    return np.array(X)

X_GRF = create_seq(scaled_data_GRF, seq_length=1) 
X_pos = create_seq(scaled_data_pos, seq_length=1) 
X_acc = create_seq(scaled_data_acc, seq_length=1) 

#convert to tensor
X_GRFtensor = torch.from_numpy(X_GRF).float() 
X_pos_tensor = torch.from_numpy(X_pos).float()
X_acc_tensor = torch.from_numpy(X_acc).float()

print(f"Input shape (GRF): {X_GRFtensor.shape}")
print(f"Input shape (position): {X_pos_tensor.shape}")
print(f"Input shape (acceleration): {X_acc_tensor.shape}")

num_time_steps__GRF = X_GRFtensor.shape[0]
num_time_steps__pos = X_pos_tensor.shape[0]
num_time_steps__acc = X_acc_tensor.shape[0]

print(f"Number of time steps (GRF): {num_time_steps__GRF}")
print(f"Number of time steps (position): {num_time_steps__pos}")
print(f"Number of time steps (acceleration): {num_time_steps__acc}")

orig_num_feature_pos = X_pos_tensor.shape[2] #number of features after fractionation
orig_num_feature_GRF = X_GRFtensor.shape[2] #number of features after fractionation
orig_num_feature_acc = X_acc_tensor.shape[2] #number of features after fractionation

print(f"Number of features after fractionation (GRF): {orig_num_feature_GRF}")
print(f"Number of features after fractionation (position): {orig_num_feature_pos}")
print(f"Number of features after fractionation (acceleration): {orig_num_feature_acc}")


#define autoencoder model and clustering model
model = GRUAutoEncoder(GRF_num_features=orig_num_feature_GRF, pos_num_features=orig_num_feature_pos, acc_num_features=orig_num_feature_acc,
                           GRF_neurons=40, pos_neurons=10, acc_neurons=20,
                           hidden_dim=32, latent_dim=4) #GRU autoencoder model

kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10) #KMeans clustering model

#loss and optimizer definitions
criterion = nn.MSELoss() #mean squared error loss for regression ( stance vs swing)

optimizer= torch.optim.Adam(model.parameters(), lr = 0.001)


#GRF training loop
epochs = 50
print("Training GRF model... ")
for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad() #clear out previous gradients
    recon_GRF, recon_pos, recon_acc, latent_total = model(X_GRFtensor, X_pos_tensor, X_acc_tensor) #forward pass

    #seperate the outputs for GRF, position, and acceleration into their respective losses
    loss_GRF = criterion(recon_GRF, X_GRFtensor) #compute loss
    loss_pos = criterion(recon_pos, X_pos_tensor) #compute loss
    loss_acc = criterion(recon_acc, X_acc_tensor) #compute loss

    #combine the losses with weights (GRF is more important than position and acceleration)
    loss = loss_GRF + loss_pos + loss_acc 
    loss.backward() #backprop
    
    optimizer.step()  #update weights

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training complete.")

model.eval() #stop training
with torch.no_grad():
    recon_GRF, recon_pos, recon_acc, latent_total = model(X_GRFtensor, X_pos_tensor, X_acc_tensor) #get latent embedding for clustering

print(f"Latent embedding shape (all modalities): {latent_total.shape} for K Means")

#apply K-Means
cluster_labels = kmeans.fit_predict(latent_total)

print(f"Clustering complete. Cluster label shape (Multimodal): {cluster_labels.shape}")

"""plot clusters"""
"""Clusters position over time of each joint"""
figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,10), sharex=True)
ax1, ax2, ax3= axes.flatten()

scatter_1 = ax1.scatter(time_vec, raw_data_pos[:,0], c=cluster_labels)
ax1.set_title("CTi clusters")
ax1.set_xlabel('time (s)')
ax1.set_ylabel('position (rads)')

scatter_2 = ax2.scatter(time_vec, raw_data_pos[:, 1], c=cluster_labels)
ax2.set_title("TrF clusters")
ax2.set_xlabel('time (s)')

scatter_3 = ax3.scatter(time_vec, raw_data_pos[:, 2], c=cluster_labels)
ax3.set_title("FTi clusters")
ax3.set_xlabel('time (s)')

cbar = figure.colorbar(scatter_3, ax=ax3, ticks=[0, 1])
cbar.ax.set_yticklabels(['Swing', 'Stance'])

figure.suptitle('LH leg Gait Classification of a Single Cycle')
plt.show()

"""Comparison of GRFs and cluster assignment"""
#plot clusters
"""Clusters position over time of each joint"""
figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,10), sharex=True)
ax4, ax5, ax6= axes.flatten()

scatter_4 = ax4.scatter(time_vec, raw_data_GRF[:, 0], c=cluster_labels)
ax4.set_title("GRF x-axis")
ax4.set_xlabel('time (s)')
ax4.set_ylabel('force (N)')

scatter_5 = ax5.scatter(time_vec, raw_data_GRF[:, 1], c=cluster_labels)
ax5.set_title("GRF y-axis")
ax5.set_xlabel('time (s)')

scatter_6 = ax6.scatter(time_vec, raw_data_GRF[:, 2], c=cluster_labels)
ax6.set_title("GRF z-axis")
ax6.set_xlabel('time (s)')
plt.show()

"""add cluster labels to csv to be our ground truths"""
"""
aligned_df = pd.read_csv("csv/gait_phase_LH_trajectories.csv")
output_labels = np.full(len(aligned_df), np.nan)
output_labels[-len(cluster_labels_GRF):] = cluster_labels_GRF
aligned_df['cluster labels'] = output_labels
aligned_df.to_csv("csv/gait_phase_LH_trajectories_wGroundTruths.csv", index=False)
print("Saved outputs successfully.") """