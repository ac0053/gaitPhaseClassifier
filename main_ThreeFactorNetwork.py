import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from models.Neurons import SpikingInput, NonSpikingOutput, NonSpikingInterneuron

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 0. Set Random Seed to Make Models Deterministic
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def deterministic(seed=42):
    # sets seeds (starting point) for python and pytorch random number generators
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.use_deterministic_algorithms(True) # makes sure algorithms used in models are deterministic

deterministic(seed=42)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Load and Pre-process Features
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fwd_df = pd.read_csv("csv_trajectory_datasets/gait_phase_LF_fwd_trajectories.csv") # forward gait
bkwd_df = pd.read_csv("csv_trajectory_datasets/gait_phase_LF_bkwds_trajectories.csv") # backward gait
#turning_df = pd.read_csv("csv_trajectory_datasets/gait_phase_LF_turn_trajectories.csv") # gait with turning

train_raw_df = pd.concat([fwd_df, bkwd_df], axis=0) # combine all datasets into one dataframe
test_raw_df = fwd_df # use forward gait as test dataset

# add Gaussian noise to tensors
def add_noise(x):
    noise = np.random.normal(loc=0, scale=0.1, size=x.shape)
    return x + noise

# load features for training and testing
#joint angles
LF_TrF_theta = add_noise(train_raw_df["LF_TrF_theta"].values)
LF_FTi_theta = add_noise(train_raw_df["LF_FTi_theta"].values)

# joint velocities
LF_TrF_vel = add_noise(train_raw_df["LF_TrF_vel"].values)
LF_FTi_vel = add_noise(train_raw_df["LF_FTi_vel"].values)

# internal joint forces
Troch_x = add_noise(train_raw_df["LF_Torque_Troch_CS_x"].values)
Troch_y = add_noise(train_raw_df["LF_Torque_Troch_CS_y"].values)
Troch_z = add_noise(train_raw_df["LF_Torque_Troch_CS_z"].values)
Tibia_x = add_noise(train_raw_df["LF_Torque_Tibia_CS_x"].values)
Tibia_y = add_noise(train_raw_df["LF_Torque_Tibia_CS_y"].values)
Tibia_z = add_noise(train_raw_df["LF_Torque_Tibia_CS_z"].values)

# normalize tensors using StandardScaler
stndscaler = StandardScaler()
LF_TrF_theta = stndscaler.fit_transform(LF_TrF_theta.reshape(-1, 1)).flatten()
LF_FTi_theta = stndscaler.transform(LF_FTi_theta.reshape(-1, 1)).flatten()
LF_TrF_vel = stndscaler.transform(LF_TrF_vel.reshape(-1, 1)).flatten()
LF_FTi_vel = stndscaler.transform(LF_FTi_vel.reshape(-1, 1)).flatten()
Troch_x = stndscaler.transform(Troch_x.reshape(-1, 1)).flatten()
Troch_y = stndscaler.transform(Troch_y.reshape(-1, 1)).flatten()
Troch_z = stndscaler.transform(Troch_z.reshape(-1, 1)).flatten()
Tibia_x = stndscaler.transform(Tibia_x.reshape(-1, 1)).flatten()
Tibia_y = stndscaler.transform(Tibia_y.reshape(-1, 1)).flatten()
Tibia_z = stndscaler.transform(Tibia_z.reshape(-1, 1)).flatten()

# combine torque and theta features into a single tensor
LF_Troch_Force = np.stack([Troch_x, Troch_y, Troch_z], axis=1)
LF_Tibia_Force = np.stack([Tibia_x, Tibia_y, Tibia_z], axis=1)
LF_thetas = np.stack([LF_TrF_theta, LF_FTi_theta], axis=1)

# sequence function adaptation 
def create_seq(data, seq_length):
    X= []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i+seq_length]) #past values
    return np.array(X)

SEQ_LENGTH = 6 
LF_thetas_seq = torch.tensor(create_seq(LF_thetas, SEQ_LENGTH), dtype=torch.float32)
LF_Troch_Force_seq = torch.tensor(create_seq(LF_Troch_Force, SEQ_LENGTH), dtype=torch.float32)
LF_Tibia_Force_seq = torch.tensor(create_seq(LF_Tibia_Force, SEQ_LENGTH), dtype=torch.float32)
  
# initialize the model's layers and neuron parameters
mag = 1.0
dt = 0.01
c_m = 1.0
theta_0 = 1.0
e_reset = 0.0
num_neurons = 100
device = torch.device('cpu')

neuron_parameters = {'c_m': c_m, 'e_rest': 0.0, 'g_m_leak': 1.0, 'e_syn': 20.0, 'dt': dt, 'R_op_range': mag, 'theta_0': theta_0, 'e_reset': e_reset}
input_layer_angles = torch.jit.script(SpikingInput(num_neurons_per_joint=num_neurons, 
                                            parameters=neuron_parameters, 
                                            device=device))
input_layer_forces = torch.jit.script(SpikingInput(num_neurons_per_joint=num_neurons, 
                                            parameters=neuron_parameters, 
                                            device=device))
nsi_layer = torch.jit.script(SpikingInput(num_neurons_per_joint=num_neurons, 
                                            parameters=neuron_parameters, 
                                            device=device))
output_layer = torch.jit.script(SpikingInput(num_neurons_per_joint=num_neurons, 
                                            parameters=neuron_parameters, 
                                            device=device))  

model = torch.nn.Sequential(
    input_layer_angles,
    input_layer_forces,
    nsi_layer,
    output_layer
)

