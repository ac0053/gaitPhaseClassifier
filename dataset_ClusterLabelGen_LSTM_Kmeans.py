import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

"""Script utilizes and LSTM autoencoder to lower feature dimensions 
and K-Means clustering to cluster joint kinematic data to either stance or swing phase"""

#initialize model
class LSTMAutoEncoder(nn.Module):
    def __init__(self, seq_length, num_features, embed_dim):
        super(LSTMAutoEncoder, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.embed_dim = embed_dim

        # encoder expects (batch, seq_len, features)
        self.encoder = nn.LSTM(input_size=num_features, #takes number of features per time step
                            hidden_size = embed_dim, #memory size
                            batch_first = True,
                            num_layers=2
                            )

        #decoder transforms latent space back to feature space
        self.decoder = nn.LSTM(input_size = embed_dim,
                               hidden_size = embed_dim,
                               batch_first = True,
                               num_layers=1)

        self.fc = nn.Linear(in_features=embed_dim, out_features=num_features) #output layer

    def forward(self,x,return_latent=False):
        _, (out_1, _) = self.encoder(x) #pass through encoder
        latent = out_1[-1]    #[time steps, embed_dim]

        if return_latent:
            return latent

        #reconstruct into feature space
        y = latent.unsqueeze(1).repeat(1, self.seq_length, 1) #turns into [time steps, seq_length, feature_dim]
        out_2, _ = self.decoder(y) #pass through decoder
        out_2 = self.fc(out_2)
        return out_2

#define hidden gait states
STATE_NAMES = {0:"Stance",
1:"Swing",
}
n_states = len(STATE_NAMES)

#load dataset
raw_df = pd.read_csv("csv/gait_phase_LH_trajectories.csv").drop(columns=['time']) #only for training autoencoder
time_vec = pd.read_csv("csv/gait_phase_LH_trajectories.csv")[['time']].values #for plotting data
raw_data = raw_df.values.astype(np.float32)
raw_tensor = torch.from_numpy(raw_data).float()
num_time_steps = raw_tensor.shape[0]
print(f"Number of time steps: {num_time_steps}")
step_size = 1
step_size = int(step_size)
print(f"Step size: {step_size}")
print(f"Input tensor shape: {raw_tensor.shape}")
num_feature = int(raw_tensor.shape[1])

#normalize data (z-norm)
scaler = StandardScaler()
normalized_tensor = scaler.fit_transform(raw_tensor)

#create sequences of data
def create_seq(data, seq_length):
    X= []
    for i in range(len(data)-seq_length+1):
        X.append(data[i:i+seq_length]) #past values
    return np.array(X)

X = create_seq(normalized_tensor, step_size)
X = torch.tensor(X, dtype=torch.float32)


#define autoencoder model and clustering model
model = LSTMAutoEncoder(seq_length=6, num_features=num_feature, embed_dim=32)
kmeans = KMeans(n_clusters=n_states, random_state=0)

#loss and optimizer definitions
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

#training loop
epochs = 50
print("Training model... ")
for epoch in range(epochs):
    model.train()

    optimizer.zero_grad() #clear out previous gradients
    outputs = model(X) #forward pass
    loss = criterion(outputs, X) #compute loss
    loss = torch.sqrt(loss) #converts MSE to RMSE
    loss.backward() #backprop
    optimizer.step()  #update weights

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("Training complete.")
model.eval() #stop training
with torch.no_grad():
    latent_embed = model(X, return_latent=True).numpy()

print(f"Latent embedding shape: {latent_embed.shape} for K Means")

#apply K-Means
cluster_labels = kmeans.fit_predict(latent_embed)
print(f"Clustering complete. Cluster label shape: {cluster_labels.shape}")

#plot clusters
"""Clusters position over time of each joint"""
figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,10), sharex=True)
ax1, ax2, ax3= axes.flatten()
scatter_1 = ax1.scatter(time_vec, raw_tensor[:, 0], c=cluster_labels)
ax1.set_title("CTi clusters")
ax1.set_xlabel('time (s)')
ax1.set_ylabel('pos (rads)')

scatter_2 = ax2.scatter(time_vec, raw_tensor[:, 3], c=cluster_labels)
ax2.set_title("TrF clusters")
ax2.set_xlabel('time (s)')
ax2.set_ylabel('pos (rads)')


scatter_3 = ax3.scatter(time_vec, raw_tensor[:, 6], c=cluster_labels)
ax3.set_title("FTi clusters")
ax3.set_xlabel('time (s)')
ax3.set_ylabel('pos (rads)')

cbar = figure.colorbar(scatter_3, ax=ax3, ticks=[0, 1])
cbar.ax.set_yticklabels(['(Stance)', '(Swing)'])

figure.suptitle('LH')
plt.show()
"""
print("Training model... ")
model.fit(normalized_tensor)
hidden_states = model.predict(normalized_tensor)
print(hidden_states)
print("Training complete. :)")

for state_num in range(n_states):
    state_mask = (hidden_states == state_num)
    if not np.any(state_mask):
        print(f"State {state_num} ({STATE_NAMES[state_num]}) was never predicted.")
        continue
    avg_CTr_acc = (raw_df.iloc[state_mask, 7]).mean()
    avg_TrF_acc = (raw_df.iloc[state_mask, 8]).mean()
    avg_FTi_acc = (raw_df.iloc[state_mask, 9]).mean()
    print(f"Discovered State {state_num} has an average acceleration of:"
      f" CTr = {avg_CTr_acc} rad/s^2"
      f" TrF = {avg_TrF_acc} rad/s^2"
      f" FTi = {avg_FTi_acc} rad/s^2"
      f"                    ")

#process via sliding windows
for start_idx in range(0, num_time_steps - window_size + 1, step_size):
    end_idx = start_idx + window_size
    window_features = normalized_tensor[start_idx:end_idx]
    print(hidden_states[start_idx:end_idx])
    latest_state = hidden_states[end_idx - 1]
    latest_state_name = STATE_NAMES.get(latest_state)
    print(f"Window [{start_idx:03d}:{end_idx:03d}] -> "
          f"Latest Inferred State: {latest_state_name}")
X_train = latest_state_name

#add cluster label to original csv and save as new csv with labels included
aligned_df.to_csv("csv/gait_phase_with_clusters.csv", index=False) #adds assigned clusters to original csv and saves as a new csv
"""


