import pandas as pd
import numpy as np

import torch
import torch.nn as nn


# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class TripletLossVAE(nn.Module):
    def __init__(self, input_dim=800, hidden_dim=512, latent_dim=20, margin=1.0, dropout=0.2):
        super(TripletLossVAE, self).__init__()
        
        # Larger encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.LeakyReLU(0.2),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim//4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim//4, latent_dim)
        
        # Larger decoder with batch normalization
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Triplet loss
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
    def compute_triplet_loss(self, z, labels):
        """Compute triplet loss for the latent representations"""
        batch_size = z.size(0)
        
        # We need at least 2 samples per class for triplet loss
        anchors = []
        positives = []
        negatives = []
        
        # Create triplets
        for i in range(batch_size):
            anchor = z[i]
            anchor_label = labels[i]
            
            # Find positive samples (same class)
            pos_mask = (labels == anchor_label) & (torch.arange(batch_size) != i)
            pos_indices = torch.where(pos_mask)[0]
            
            # Find negative samples (different class)
            neg_mask = (labels != anchor_label)
            neg_indices = torch.where(neg_mask)[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                # Randomly select one positive and one negative
                pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))]
                neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,))]
                
                anchors.append(anchor.unsqueeze(0))
                positives.append(z[pos_idx])
                negatives.append(z[neg_idx])
        
        if len(anchors) > 0:
            anchors = torch.cat(anchors)
            positives = torch.cat(positives)
            negatives = torch.cat(negatives)
            
            return self.triplet_loss(anchors, positives, negatives)
        else:
            # Return zero loss if we can't form triplets
            return torch.tensor(0.0).to(z.device)

def improved_vae_loss(recon_x, x, mu, logvar, beta=0.001):
    """Improved VAE loss with adjustable beta and better reconstruction loss"""
    # Use smooth L1 loss for better gradient behavior
    recon_loss = nn.SmoothL1Loss()(recon_x, x)
    
    # KL divergence
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #print('KL:', kld_loss)
    return recon_loss + beta * kld_loss, kld_loss