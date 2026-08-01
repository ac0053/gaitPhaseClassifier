import torch
import torch.nn as nn
from models.RangeFractionation import GaussianRangeFractionation

# feed forward autoencoder will still utilize range fractionated inputs
class FeedFwdAutoencoder(nn.Module): # autoencoder utilizing linear layers for compression rather than GRU
    def __init__(self, theta_num_features, vel_num_features, theta_neurons, vel_neurons, hidden_dim, latent_dim, seq_length,drop_rate=0.2):
        super(FeedFwdAutoencoder, self).__init__()
        self.theta_num_features = theta_num_features
        self.vel_num_features = vel_num_features

        self.theta_neurons = theta_neurons
        self.vel_neurons = vel_neurons
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.drop_rate = drop_rate
        
        # seperate the input data into a range of neurons (populations) with seperate streams for each feature, then concatenate the streams into a single input for the linear layers
        self.theta_fractionation = GaussianRangeFractionation(num_neurons=theta_neurons, min_value=-1.0, max_value=1.0)  
        self.vel_fractionation = GaussianRangeFractionation(num_neurons=vel_neurons, min_value=-1.0, max_value=1.0)  
                
        fractionated_theta_input_dim = self.theta_neurons * self.theta_num_features  
        fractionated_vel_input_dim = self.vel_neurons * self.vel_num_features 
        total_fractionated_input_dim = fractionated_theta_input_dim + fractionated_vel_input_dim
        self.total_fractionated_input_dim = total_fractionated_input_dim 

        # define encoder and decoder as sequential model with linear layers and RELU activation functions
        # linear compression (encoder)
        self.encoder = nn.Sequential(
            nn.Linear(self.total_fractionated_input_dim, hidden_dim),
            nn.ReLU(), # to introduce non-linearity
            nn.Dropout(drop_rate), # for monte carlo dropout to estimate uncertainty in latent embedding
            nn.Linear(hidden_dim, latent_dim)
        )

        # liner reconstruction (decoder)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # to match encoder dimensions from gru autoencoder
            nn.ReLU(),
            nn.Dropout(drop_rate), # for monte carlo dropout to estimate uncertainty in reconstruction
            nn.Linear(hidden_dim, self.total_fractionated_input_dim) # reconstruct back to total dimensions
        )

        self.theta_fc = nn.Linear(self.total_fractionated_input_dim, self.theta_num_features)  # output layer for joint angles
        self.vel_fc = nn.Linear(self.total_fractionated_input_dim, self.vel_num_features)  # output layer for velocity

    def forward(self, theta_x, vel_x):
        thetafractionated_x = self.theta_fractionation(theta_x)  
        thetafractionated_x = self.theta_fractionation(theta_x)  
        velfractionated_x = self.vel_fractionation(vel_x)  

       #concatenate fractionated inputs
        total_fractionated_x = torch.cat((thetafractionated_x, velfractionated_x), dim=2)  #concatenate along feature dimension
        batch_size, seq_length, _ = total_fractionated_x.shape

        # encoder
        latent_embed = nn.functional.dropout(self.encoder(total_fractionated_x), p=self.drop_rate, training=True)  # apply dropout for uncertainty estimation
        # latent to probability
        probs = torch.sigmoid(latent_embed)

        # decoder
        recon = self.decoder(latent_embed)
        recon_total = nn.functional.dropout(recon, p=self.drop_rate, training=True)  # apply dropout for uncertainty estimation

        theta_recon = self.theta_fc(recon_total)  #reconstructed theta
        vel_recon = self.vel_fc(recon_total)  #reconstructed vel

        return theta_recon, vel_recon, probs
