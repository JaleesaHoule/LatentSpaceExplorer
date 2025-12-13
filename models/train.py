#!/usr/bin/env python3
"""
Main training script for VAE with triplet loss.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from datetime import datetime

from models.triplet_vae_basic import TripletLossVAE
from utils.data_processing import TrajectoryDataset, prepare_data
from utils.diagnostics import plot_training_progress


def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE with triplet loss')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=20, help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.001, help='KL loss weight')
    parser.add_argument('--triplet_weight', type=float, default=0.1, help='Triplet loss weight')
    parser.add_argument('--margin', type=float, default=1.0, help='Triplet margin')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()
'''

def train_vae_with_triplet(model, train_loader, test_loader, epochs=200, 
                          triplet_weight=0.1, beta=0.001, lr=1e-4):
    """Train the VAE model with triplet loss"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    triplet_losses = []
    val_triplet_losses = []
    kl_losses = []
    val_kl_losses = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        total_triplet_loss = 0
        total_vae_loss = 0
        total_kl_loss = 0
        
        for batch_idx, (data, labels, obj_ids) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar, z = model(data)
            
            # Reconstruction loss
            recon_loss = nn.SmoothL1Loss()(recon_batch, data)
            
            # KL divergence
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Triplet loss
            triplet_loss = model.compute_triplet_loss(z, labels)
            
            # Combined loss
            vae_loss = recon_loss + beta * kld_loss
            total_loss = vae_loss + triplet_weight * triplet_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += total_loss.item()
            total_vae_loss += vae_loss.item()
            total_triplet_loss += triplet_loss.item()
            total_kl_loss += kld_loss.item()
        
        # Validation
        model.eval()
        total_val_loss = 0
        total_val_triplet_loss = 0
        total_val_vae_loss = 0
        total_val_kl_loss = 0
        
        with torch.no_grad():
            for data, labels, obj_ids in test_loader:
                data, labels = data.to(device), labels.to(device)
                recon_batch, mu, logvar, z = model(data)
                
                recon_loss = nn.SmoothL1Loss()(recon_batch, data)
                kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                triplet_loss = model.compute_triplet_loss(z, labels)
                
                vae_loss = recon_loss + beta * kld_loss
                total_loss = vae_loss + triplet_weight * triplet_loss
                
                total_val_loss += total_loss.item()
                total_val_vae_loss += vae_loss.item()
                total_val_triplet_loss += triplet_loss.item()
                total_val_kl_loss += kld_loss.item()
        
        # Calculate averages
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_triplet_loss = total_triplet_loss / len(train_loader.dataset)
        avg_vae_loss = total_vae_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        
        avg_val_loss = total_val_loss / len(test_loader.dataset)
        avg_val_triplet_loss = total_val_triplet_loss / len(test_loader.dataset)
        avg_val_vae_loss = total_val_vae_loss / len(test_loader.dataset)
        avg_val_kl_loss = total_val_kl_loss / len(test_loader.dataset)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        triplet_losses.append(avg_triplet_loss)
        val_triplet_losses.append(avg_val_triplet_loss)
        kl_losses.append(avg_kl_loss)
        val_kl_losses.append(avg_val_kl_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}:')
            print(f'  Train - Total: {avg_train_loss:.6f}, VAE: {avg_vae_loss:.6f}, '
                  f'Triplet: {avg_triplet_loss:.6f}, KL: {avg_kl_loss:.6f}')
            print(f'  Val   - Total: {avg_val_loss:.6f}, VAE: {avg_val_vae_loss:.6f}, '
                  f'Triplet: {avg_val_triplet_loss:.6f}, KL: {avg_val_kl_loss:.6f}')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'triplet_losses': triplet_losses,
        'val_triplet_losses': val_triplet_losses,
        'kl_losses': kl_losses,
        'val_kl_losses': val_kl_losses
    }
'''

            
def train_vae_with_triplet(model, train_loader, test_loader, epochs=200, triplet_weight=0.1, beta=0.001):
    """Train the VAE model with triplet loss - PROPER VALIDATION"""
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    triplet_losses = []
    val_triplet_losses = []  # Track validation triplet loss too
    val_total_losses = []    # Track combined validation loss
    kl_losses=[]
    val_kl_losses=[]
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        total_triplet_loss = 0
        total_vae_loss = 0
        total_kl_loss = 0
        for batch_idx, (data, labels,obj) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar, z = model(data)
            
            # Compute VAE loss
            vae_loss, kl_loss = improved_vae_loss(recon_batch, data, mu, logvar, beta=beta)
            
            # Compute triplet loss
            triplet_loss = model.compute_triplet_loss(z, labels)
            
            # Combined loss
            total_loss = vae_loss + triplet_weight * triplet_loss
            
            total_loss.backward()
            
            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += total_loss.item()
            total_vae_loss += vae_loss.item()
            total_triplet_loss += triplet_loss.item()
            total_kl_loss += kl_loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        avg_triplet_loss = total_triplet_loss / len(train_loader.dataset)
        avg_vae_loss = total_vae_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)
        
        train_losses.append(avg_train_loss)
        triplet_losses.append(avg_triplet_loss)
        kl_losses.append(avg_kl_loss)
        
        # VALIDATION - Include triplet loss!
        model.eval()
        total_val_loss = 0
        total_val_triplet_loss = 0
        total_val_vae_loss = 0
        total_val_kl_loss = 0
        with torch.no_grad():
            for data, labels, obj in test_loader:
                recon_batch, mu, logvar, z = model(data)
                
                # Compute both losses for validation
                vae_loss, kl_loss = improved_vae_loss(recon_batch, data, mu, logvar, beta=beta)
                triplet_loss = model.compute_triplet_loss(z, labels)
                
                # Combined validation loss (same weighting as training)
                combined_val_loss = vae_loss + triplet_weight * triplet_loss
                
                total_val_loss += combined_val_loss.item()
                total_val_triplet_loss += triplet_loss.item()
                total_val_vae_loss += vae_loss.item()
                total_val_kl_loss += kl_loss.item()
                
        avg_val_loss = total_val_loss / len(test_loader.dataset)
        avg_val_triplet_loss = total_val_triplet_loss / len(test_loader.dataset)
        avg_val_vae_loss = total_val_vae_loss / len(test_loader.dataset)
        avg_val_kl_loss = total_val_kl_loss / len(test_loader.dataset)
        
        val_losses.append(avg_val_loss)
        val_triplet_losses.append(avg_val_triplet_loss)
        val_total_losses.append(avg_val_loss)
        val_kl_losses.append(avg_val_kl_loss)
        
        # Use COMBINED loss for scheduler
        scheduler.step(avg_val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}:')
            print(f'  Train - Total: {avg_train_loss:.6f}, VAE: {avg_vae_loss:.6f}, Triplet: {avg_triplet_loss:.6f}, KL: {avg_kl_loss:.6f}')
            print(f'  Val   - Total: {avg_val_loss:.6f}, VAE: {avg_val_vae_loss:.6f}, Triplet: {avg_val_triplet_loss:.6f}, KL: {avg_val_kl_loss:.6f} ')
            print('')
    
    return train_losses, val_losses, triplet_losses, val_triplet_losses, kl_losses, val_kl_losses


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load and prepare data
    print(f"Loading data from {args.data}...")
    trajectories, labels, obj_ids = prepare_data(args.data)
    input_dim = trajectories.shape[1] * trajectories.shape[2]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, obj_train, obj_test = train_test_split(
        trajectories, labels, obj_ids, test_size=0.3, random_state=args.seed, stratify=labels
    )
    
    # Create datasets
    train_dataset = TrajectoryDataset(X_train, y_train, obj_train, normalization_type='zero_mean')
    test_dataset = TrajectoryDataset(X_test, y_test, obj_test, normalization_type='zero_mean')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print(f"Input dimension: {input_dim}")
    model = TripletLossVAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        margin=args.margin
    )
    
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("\nStarting training...")
    metrics = train_vae_with_triplet(
        model, train_loader, test_loader,
        epochs=args.epochs,
        triplet_weight=args.triplet_weight,
        beta=args.beta,
        lr=args.learning_rate
    )
    
    # Save model and results
    model_path = os.path.join(args.output_dir, f"model_{timestamp}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'metrics': metrics
    }, model_path)
    
    # Save training history
    history_path = os.path.join(args.output_dir, f"training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot training progress
    plot_path = os.path.join(args.output_dir, f"training_plot_{timestamp}.png")
    plot_training_progress(metrics, save_path=plot_path)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Training plot saved to: {plot_path}")


if __name__ == '__main__':
    main()