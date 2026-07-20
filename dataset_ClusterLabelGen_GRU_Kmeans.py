import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random


"""Script utilizes a range-fractionated GRU autoencoder to lower feature dimensions 
and K-Means clustering to cluster joint kinematic and GRF data to either stance or swing phase"""
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 0. Set random seed to make models deterministic
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def deterministic(seed=42):
    #sets seeds (starting point) for python and pytorch random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

deterministic(seed=42)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Define Models
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# range fractionation
class GaussianRangeFractionation(nn.Module): # to be used before autoencoder to fractionate the input data into a range of neurons (acts as encoder)
    def __init__(self, num_neurons, min_value, max_value, sigma=0.1):
        super(GaussianRangeFractionation, self).__init__()
        self.num_neurons = num_neurons
        self.min_value = min_value
        self.max_value = max_value
        self.sigma = sigma

        centers = torch.linspace(min_value, max_value, num_neurons)  # centers of the Gaussian functions
        self.register_buffer('centers',centers) # register centers as a buffer (fixed tensor that is not a parameter)

        self.sigma_param = torch.full((num_neurons,), sigma)  # width of bell curves (1d tensor of size num_neurons)

    def forward(self, x):
        # expected input shape: [batch_size, seq_len, num_features]
        batch_size, seq_len, num_features = x.shape
        
        # expand input dimensions for broadcasting: [batch, seq_len, features, 1]
        x_expanded = x.unsqueeze(-1)
        
        # reshape centers & sigmas for broadcasting: [1, 1, 1, num_neurons]
        mu = self.centers.view(1, 1, 1, -1)
        sigma = self.sigma_param.view(1, 1, 1, -1)
        
        # compute Gaussian RBF Activation
        squared_diff = (x_expanded - mu) ** 2
        variance = 2.0 * (sigma ** 2)
        activations = torch.exp(-squared_diff / variance)
        
        # [batch_size, seq_len, num_features * num_neurons]
        return activations.view(batch_size, seq_len, num_features*self.num_neurons)

# autoencoder
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
        self.GRF_fractionation = GaussianRangeFractionation(num_neurons=GRF_neurons, min_value=-1.0, max_value=1.0)  
        self.pos_fractionation = GaussianRangeFractionation(num_neurons=pos_neurons, min_value=-1.0, max_value=1.0)  
        self.acc_fractionation = GaussianRangeFractionation(num_neurons=acc_neurons, min_value=-1.0, max_value=1.0)  
        
        fractionated_GRF_input_dim = self.GRF_neurons * self.GRF_num_features 
        fractionated_pos_input_dim = self.pos_neurons * self.pos_num_features  
        fractionated_acc_input_dim = self.acc_neurons * self.acc_num_features 
        total_fractionated_input_dim = fractionated_GRF_input_dim + fractionated_pos_input_dim + fractionated_acc_input_dim
        self.total_fractionated_input_dim = total_fractionated_input_dim 

        """for encoder function, we will use a GRU to encode the fractionated input data into a latent space of dimension embed_dim"""
        #shared encoder, expects [batch, seq_length, features]
        self.encoder_gru = nn.GRU(input_size=total_fractionated_input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1)

        #latent embedding layer to reduce the hidden state to a lower dimension
        self.hidden2latten = nn.Linear(hidden_dim, latent_dim) 
        
        #map latent embedding back to hidden dimension for decoder
        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)

        #shared decoder
        self.decoder_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1)

        self.outputGRF = nn.Linear(hidden_dim, self.GRF_num_features)  #output layer for GRF
        self.outputPos = nn.Linear(hidden_dim, self.pos_num_features)  #output layer for position
        self.outputAcc = nn.Linear(hidden_dim, self.acc_num_features)  #output layer for acceleration

    def forward(self, GRF_x, pos_x, acc_x):
        #fractionate input data
        #seq should be same value for all inputs
        GRFfractionated_x = self.GRF_fractionation(GRF_x)  
        posfractionated_x = self.pos_fractionation(pos_x)  
        accfractionated_x = self.acc_fractionation(acc_x)  

        #concatenate fractionated inputs
        total_fractionated_x = torch.cat((GRFfractionated_x, posfractionated_x, accfractionated_x), dim=2)  #concatenate along feature dimension
        batch_size, seq_length, _ = total_fractionated_x.shape

        #encoder
        _, hidden = self.encoder_gru(total_fractionated_x)  #hidden shape: (num_layers, batch, hidden_dim)
        hidden = hidden[-1]  #take the last layer's hidden state (shape: (batch, hidden_dim))
        latent_embed = self.hidden2latten(hidden)  #project to latent space
        latent_embed_expanded = latent_embed.unsqueeze(1).repeat(1, seq_length, 1)  #expand latent embedding to match sequence length *add on to this bestie

        #decoder
        latent_to_hidden = self.latent2hidden(latent_embed_expanded)  #map latent embedding back to hidden dimension
        decoder_output, _ = self.decoder_gru(latent_to_hidden)  #decoder output

        #seperate the outputs for GRF, position, and acceleration
        GRF_recon = self.outputGRF(decoder_output)  #reconstructed GRF
        pos_recon = self.outputPos(decoder_output)  #reconstructed position
        acc_recon = self.outputAcc(decoder_output)  #reconstructed acceleration

        return GRF_recon, pos_recon, acc_recon, latent_embed

#kmeans
class KMeans(nn.Module):
    def __init__(self, num_clusters):
        super(KMeans, self).__init__()
        self.num_clusters = num_clusters
        self.centroids = None

    def initialize_centroids(self, latent_data):
        # randomly initializes centroids from the data points
        num_samples = latent_data.size(0)
        # creates 1d tensor containing random permutation of integers from 0 to num_samples - 1 and grabs first k indices
        random_indices = torch.randperm(num_samples)[:self.num_clusters]
        # use random indices to get k random samples
        self.centroids = latent_data[random_indices].clone().detach()

    def forward(self, latent_data, num_iterations):
        if self.centroids is None:
            self.initialize_centroids(latent_data)

        centroids = self.centroids.clone()
        for _ in range(num_iterations):
            # calculate distances: output shape [N, K]
            distances = torch.cdist(latent_data, self.centroids) # euclidean norm
            
            # assign each point to the closest centroid
            labels = torch.argmin(distances, dim=1)
            
            new_centroids = []
            for i in range(self.num_clusters):
                # find all points belonging to cluster i
                cluster_points = latent_data[labels == i]
                
                if len(cluster_points) > 0:
                    # take the mean to update centroid
                    new_center = torch.mean(cluster_points, dim=0)
                else:
                    # handle empty clusters by re-initializing randomly
                    new_center = centroids[i] 
                    
                new_centroids.append(new_center)
                
            centroids = torch.stack(new_centroids)

        self.centroids = centroids.detach()
        return self.centroids, labels
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. load and preprocess data
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
raw_df = pd.read_csv("csv/gait_phase_LH_trajectories.csv").drop(columns=['time']) #only for training autoencoder, already indexed by timestep
time_vec = pd.read_csv("csv/gait_phase_LH_trajectories.csv")[['time']].values #for plotting data
n_states = 2 #for clustering stance and swing phases

#preprocess data
def scale_dataframe(dataframe): #seperates dataframe into different modalities and z-norms them
    scaler = StandardScaler()
    raw_data = dataframe.values.astype(np.float32)

    raw_data_GRF = raw_data[:, :3]  #extract GRF data
    scaled_data_GRF = scaler.fit_transform(raw_data_GRF)

    raw_data_pos = raw_data[:, [3,5,7]]  #extract joint kinematics data
    scaled_data_pos = scaler.fit_transform(raw_data_pos)

    raw_data_acc = raw_data[:, [4,6,8]]  #extract joint kinematics data
    scaled_data_acc = scaler.fit_transform(raw_data_acc)
    return scaled_data_GRF, scaled_data_pos, scaled_data_acc, raw_data_GRF, raw_data_pos, raw_data_acc

#sequence function adaptation for autoencoder input
def create_seq(data, seq_length):
    X= []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i+seq_length]) #past values
    return np.array(X)

scaled_GRF, scaled_pos, scaled_acc, raw_data_GRF, raw_data_pos, raw_data_acc = scale_dataframe(raw_df) #raw data to be used for plotting

SEQ_LENGTH = 6 #best if sequence length is a diviser of the number of time steps *why tho?
X_GRF = create_seq(scaled_GRF, seq_length=SEQ_LENGTH) 
X_pos = create_seq(scaled_pos, seq_length=SEQ_LENGTH) 
X_acc = create_seq(scaled_acc, seq_length=SEQ_LENGTH) 

#convert to tensor
X_GRFtensor = torch.from_numpy(X_GRF).float() 
X_pos_tensor = torch.from_numpy(X_pos).float()
X_acc_tensor = torch.from_numpy(X_acc).float()

print(f"Input shape (GRF): {X_GRFtensor.shape}")
print(f"Input shape (position): {X_pos_tensor.shape}")
print(f"Input shape (acceleration): {X_acc_tensor.shape}")

orig_num_feature_pos = X_pos_tensor.shape[2] 
orig_num_feature_GRF = X_GRFtensor.shape[2] 
orig_num_feature_acc = X_acc_tensor.shape[2] 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Initiazlize Autoencoder
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model = GRUAutoEncoder(GRF_num_features=orig_num_feature_GRF, pos_num_features=orig_num_feature_pos, acc_num_features=orig_num_feature_acc,
                           GRF_neurons=3200, pos_neurons=1600, acc_neurons=1600,
                           hidden_dim=64, latent_dim=1) #GRU autoencoder model *add noise to data?

#loss and optimizer definitions
criterion = nn.MSELoss() #mean squared error loss for data reconstruction

optimizer= torch.optim.Adam(model.parameters(), lr = 0.01)

#define loss weights (GRF more important than kinematics)
ALPHA = 3.0  #weight for GRF
BETA = 1.0   #weight for Position
GAMMA = 1.0  #weight for Acceleration

dataset = TensorDataset(X_GRFtensor, X_pos_tensor, X_acc_tensor)
dataloader = DataLoader(dataset, batch_size=64)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. training loop
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
epochs = 50
print("Training GRF model... ")
for epoch in range(epochs):
    model.train()
    
    epoch_loss = 0.0
    
    for batch_GRF, batch_pos, batch_acc in dataloader:
        optimizer.zero_grad() 
        
        #forward pass on the mini-batch
        recon_GRF, recon_pos, recon_acc, _ = model(batch_GRF, batch_pos, batch_acc) 
        
        # compute individual losses
        loss_GRF = criterion(recon_GRF, batch_GRF) 
        loss_pos = criterion(recon_pos, batch_pos) 
        loss_acc = criterion(recon_acc, batch_acc) 
        
        # weighted loss combination
        loss = (ALPHA * loss_GRF) + (BETA * loss_pos) + (GAMMA * loss_acc)
        
        loss.backward() # backprop
        optimizer.step()  # update parameter

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training complete.")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 5. Testing
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model.eval() #stop training
with torch.no_grad():
    recon_GRF, recon_pos, recon_acc, latent_total = model(X_GRFtensor, X_pos_tensor, X_acc_tensor) #get latent embedding for clustering

print(f"Latent embedding shape (all modalities): {latent_total.shape} for K Means")

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. Initialize KMeans model
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
kmeans = KMeans(num_clusters=2)
centroids, labels = kmeans.forward(latent_data=latent_total, num_iterations=100)
labels = labels[:-1] #will have indexing error for plotting if this is not here

print(f"Clustering complete. Cluster label shape (Multimodal): {labels.shape}")
print(f"Centroid shape: {centroids.shape}")

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. plot data
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
aligned_time = time_vec[SEQ_LENGTH:]
"""plot range fractionated input """
range_frac_inst = GaussianRangeFractionation(num_neurons=1, min_value=-1.0, max_value=1.0)
pos_frac = range_frac_inst(X_pos_tensor).detach().cpu().numpy()

figure, axes = plt.subplots(nrows=1, ncols=4, figsize=(20,25))
ax7, ax8, ax9, ax10 = axes.flatten()

ax7.plot(pos_frac[:, :, 0])
ax7.set_title("CTi Joint Fractionation")
ax7.set_xlabel('time steps (per 2 ms)')
ax7.set_ylabel('magnitude')

ax8.plot(pos_frac[:, :, 1])
ax8.set_title("TrF Joint Fractionation")
ax8.set_xlabel('time steps (per 2 ms)')

ax9.plot(pos_frac[:, :, 2])
ax9.set_title("FTi Joint Fractionation")
ax9.set_xlabel('time steps (per 2 ms)')

ax10.plot(pos_frac[:, -1, :])
ax10.set_title("All 3 Joints Fractionation")
ax10.set_xlabel('time steps (per 2 ms)')
figure.suptitle("Range Fractionated Joint Angles")
plt.show()


"""angles over time of each joint"""
figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,10), sharex=True)
ax1, ax2, ax3= axes.flatten()

scatter_1 = ax1.scatter(aligned_time, raw_data_pos[SEQ_LENGTH:,0], c=labels)
ax1.set_title("CTi angles")
ax1.set_xlabel('time (s)')
ax1.set_ylabel('joint angle (rads)')

scatter_2 = ax2.scatter(aligned_time, raw_data_pos[SEQ_LENGTH:, 1], c=labels)
ax2.set_title("TrF angles")
ax2.set_xlabel('joint angle (s)')

scatter_3 = ax3.scatter(aligned_time, raw_data_pos[SEQ_LENGTH:, 2], c=labels)
ax3.set_title("FTi angles")
ax3.set_xlabel('time (s)')

cbar = figure.colorbar(scatter_3, ax=ax3, ticks=[0, 1])
cbar.ax.set_yticklabels(['Stance', 'Swing'])

figure.suptitle('LH leg Gait Classification of a Single Cycle')
plt.show()

"""Comparison of GRFs and cluster assignment"""
#plot clusters
figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,10), sharex=True)
ax4, ax5, ax6= axes.flatten()

scatter_4 = ax4.scatter(aligned_time, raw_data_GRF[SEQ_LENGTH:, 0], c=labels, cmap='plasma')
ax4.set_title("GRF x-axis")
ax4.set_xlabel('time (s)')
ax4.set_ylabel('force (N)')

scatter_5 = ax5.scatter(aligned_time, raw_data_GRF[SEQ_LENGTH:, 1], c=labels, cmap='plasma')
ax5.set_title("GRF y-axis")
ax5.set_xlabel('time (s)')

scatter_6 = ax6.scatter(aligned_time, raw_data_GRF[SEQ_LENGTH:, 2], c=labels, cmap='plasma')
ax6.set_title("GRF z-axis")
ax6.set_xlabel('time (s)')
cbar = figure.colorbar(scatter_6, ax=ax6, ticks=[0, 1])
cbar.ax.set_yticklabels(['Stance', 'Swing'])
plt.show()

"""
for testing, maybe try different walking directions *with* noise?
try with just GRF z force and no other information
instead of acc, try vel 

"""