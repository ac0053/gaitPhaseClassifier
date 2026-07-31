import torch
import torch.nn as nn
from models.RangeFractionation import GaussianRangeFractionation

# feed forward autoencoder will still utilize range fractionated inputs
class FeedFwdAutoencoder(nn.Module): # autoencoder utilizing linear layers for compression rather than GRU
    def __init__(self, GRF_num_features, theta_num_features, vel_num_features, GRF_neurons, theta_neurons, vel_neurons, hidden_dim, latent_dim, seq_length):
        super(FeedFwdAutoencoder, self).__init__()
        self.GRF_num_features = GRF_num_features
        self.theta_num_features = theta_num_features
        self.vel_num_features = vel_num_features
        self.GRF_neurons = GRF_neurons
        self.theta_neurons = theta_neurons
        self.vel_neurons = vel_neurons
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        
        # seperate the input data into a range of neurons (populations) with seperate streams for each feature, then concatenate the streams into a single input for the linear layers
        self.GRF_fractionation = GaussianRangeFractionation(num_neurons=GRF_neurons, min_value=-1.0, max_value=1.0)  
        self.theta_fractionation = GaussianRangeFractionation(num_neurons=theta_neurons, min_value=-1.0, max_value=1.0)  
        self.vel_fractionation = GaussianRangeFractionation(num_neurons=vel_neurons, min_value=-1.0, max_value=1.0)  
                
        fractionated_GRF_input_dim = self.GRF_neurons * self.GRF_num_features 
        fractionated_theta_input_dim = self.theta_neurons * self.theta_num_features  
        fractionated_vel_input_dim = self.vel_neurons * self.vel_num_features 
        total_fractionated_input_dim = fractionated_GRF_input_dim + fractionated_theta_input_dim + fractionated_vel_input_dim
        self.total_fractionated_input_dim = total_fractionated_input_dim 

        # define encoder and decoder as sequential model with linear layers and RELU activation functions
        # linear compression (encoder)
        self.encoder = nn.Sequential(
            nn.Linear(self.total_fractionated_input_dim, hidden_dim),
            nn.ReLU(), # to introduce non-linearity
            nn.Linear(hidden_dim, latent_dim)
        )

        # liner reconstruction (decoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # to match encoder dimensions from gru autoencoder
            nn.Linear(hidden_dim, self.total_fractionated_input_dim) # reconstruct back to total dimensions
        )

        self.GRF_fc = nn.Linear(self.total_fractionated_input_dim, self.GRF_num_features)  # output layer for GRF
        self.theta_fc = nn.Linear(self.total_fractionated_input_dim, self.theta_num_features)  # output layer for thetaition
        self.vel_fc = nn.Linear(self.total_fractionated_input_dim, self.vel_num_features)  # output layer for veleleration

    def forward(self, GRF_x, theta_x, vel_x):
        GRFfractionated_x = self.GRF_fractionation(GRF_x)  
        thetafractionated_x = self.theta_fractionation(theta_x)  
        velfractionated_x = self.vel_fractionation(vel_x)  

       #concatenate fractionated inputs
        total_fractionated_x = torch.cat((GRFfractionated_x, thetafractionated_x, velfractionated_x), dim=2)  #concatenate along feature dimension
        batch_size, seq_length, _ = total_fractionated_x.shape

        # flatten fractionated_x to 2d tensor for k means clustering (shape becomes [batch, seq_length*num_features])
        flat_x = total_fractionated_x.view(batch_size, -1) 

        # encoder
        latent_embed = self.encoder(total_fractionated_x)

        # latent to probability
        probs = torch.sigmoid(latent_embed)

        # decoder
        recon_flat = self.decoder(latent_embed)
        recon_total = recon_flat.reshape(batch_size, seq_length, self.total_fractionated_input_dim) # changes back into [batch, seq_length, num_features] shape to compare to orignal data during training

        GRF_recon = self.GRF_fc(recon_total)  #reconstructed GRF
        theta_recon = self.theta_fc(recon_total)  #reconstructed theta
        vel_recon = self.vel_fc(recon_total)  #reconstructed vel

        return GRF_recon, theta_recon, vel_recon, probs
