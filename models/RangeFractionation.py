import torch
import torch.nn as nn

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