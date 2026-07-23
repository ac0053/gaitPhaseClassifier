import torch
import torch.nn as nn

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
            # calculate distances between latent data and current centroids: output shape [N, K]
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
                
            centroids = torch.stack(new_centroids) # concatenates sequence of tensors along a new dimension

        self.centroids = centroids.detach()
        return self.centroids, labels