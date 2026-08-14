import torch
import torch.nn as nn
from models.RangeFractionation import GaussianRangeFractionation

class GRUAutoEncoder(nn.Module):
    def __init__(self, theta_num_features, vel_num_features, lin_acc_num_features,
                theta_neurons, vel_neurons, lin_acc_neurons,
                hidden_dim, latent_dim, drop_rate=0.2):
        super(GRUAutoEncoder, self).__init__()
        self.theta_num_features = theta_num_features
        self.vel_num_features = vel_num_features
        self.lin_acc_num_features = lin_acc_num_features
        self.theta_neurons = theta_neurons
        self.vel_neurons = vel_neurons
        self.lin_acc_neurons = lin_acc_neurons
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.drop_rate = drop_rate

        # encoder expects (batch, seq_len, features)
        #seperate the input data into a range of neurons (populations) with seperate streams for each feature, then concatenate the streams into a single input for the GRU
        #self.GRF_fractionation = GaussianRangeFractionation(num_neurons=GRF_neurons, min_value=-1.0, max_value=1.0)  
        self.theta_fractionation = GaussianRangeFractionation(num_neurons=theta_neurons, min_value=-1.0, max_value=1.0)  
        self.vel_fractionation = GaussianRangeFractionation(num_neurons=vel_neurons, min_value=-1.0, max_value=1.0)  
        self.lin_acc_fractionation = GaussianRangeFractionation(num_neurons=lin_acc_neurons, min_value=-1.0, max_value=1.0)
        
        #fractionated_GRF_input_dim = self.GRF_neurons * self.GRF_num_features 
        fractionated_theta_input_dim = self.theta_neurons * self.theta_num_features  
        fractionated_vel_input_dim = self.vel_neurons * self.vel_num_features 
        fractionated_lin_acc_input_dim = self.lin_acc_neurons * self.lin_acc_num_features
        total_fractionated_input_dim = fractionated_theta_input_dim + fractionated_vel_input_dim + fractionated_lin_acc_input_dim
        self.total_fractionated_input_dim = total_fractionated_input_dim 

        """for encoder function, we will use a GRU to encode the fractionated input data into a latent space of dimension latent_dim"""
        #shared encoder, expects [batch, seq_length, features]
        self.encoder_gru = nn.GRU(input_size=total_fractionated_input_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1)

        #latent embedding layer to reduce the hidden state to a lower dimension
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim) 
        
        #map latent embedding back to hidden dimension for decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
    
        #shared decoder
        self.decoder_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1)

        self.theta_fc = nn.Linear(hidden_dim, self.theta_num_features)  #output layer for joint angles
        self.vel_fc = nn.Linear(hidden_dim, self.vel_num_features)  #output layer for velocity
        self.lin_acc_fc = nn.Linear(hidden_dim, self.lin_acc_num_features) # output layer for end effector's vertical linear acceleration

    def forward(self, theta_x, vel_x, lin_eff_acc):
        # fractionate input data
        # seq should be same value for all inputs
    
        thetafractionated_x = self.theta_fractionation(theta_x)  
        velfractionated_x = self.vel_fractionation(vel_x)  
        eff_lin_acc_fractionated = self.lin_acc_fractionation(lin_eff_acc)

        # concatenate fractionated inputs
        total_fractionated_x = torch.cat((thetafractionated_x, velfractionated_x, eff_lin_acc_fractionated), dim=2)  #concatenate along feature dimension
        batch_size, seq_length, _ = total_fractionated_x.shape

        # encoder
        _, hidden = self.encoder_gru(total_fractionated_x)  #hidden shape: (num_layers, batch, hidden_dim)

        # take the last layer's hidden state (shape: (batch, hidden_dim)) and apply dropout for uncertainty estimation
        hidden = nn.functional.dropout(hidden[-1], p=self.drop_rate, training=True)  # apply dropout for uncertainty estimation
        latent_embed = self.encoder_fc(hidden)  # project to latent space


        # assign sigmoid to 1D latent data to get probabilities
        probs = torch.sigmoid(latent_embed)

        latent_embed_expanded = latent_embed.unsqueeze(1).repeat(1, seq_length, 1)  # expand latent embedding to match sequence length for reconstruction

        #decoder
        latent_to_hidden = self.decoder_fc(latent_embed_expanded)  # map latent embedding back to hidden dimension
        decoder_output, _ = self.decoder_gru(latent_to_hidden)  #decoder output

        # apply dropout for uncertainty estimation
        decoder_output = nn.functional.dropout(decoder_output, p=self.drop_rate, training=True)  #apply dropout for uncertainty estimation

        #seperate the outputs for GRF, joint angles, and velocity
        theta_recon = self.theta_fc(decoder_output)  #r econstructed joint angles
        vel_recon = self.vel_fc(decoder_output)  # reconstructed velocity
        lin_acc_recon = self.lin_acc_fc(decoder_output) # reconstructed vertical linear accelearation of end effector
        return theta_recon, vel_recon, lin_acc_recon, probs