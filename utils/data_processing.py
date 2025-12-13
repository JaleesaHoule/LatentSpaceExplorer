"""
Data processing utilities for trajectory data.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories, labels, obj_ids=None, colors=None, normalization_type='zero_mean'):
        trajectories_flat = trajectories.reshape(trajectories.shape[0], -1)
        
        if normalization_type == 'zero_mean':
            self.trajectories = self.normalize_zero_mean(trajectories_flat)
        elif normalization_type == 'minmax':
            self.trajectories = self.normalize_minmax(trajectories_flat)
        else:
            raise ValueError("normalization_type must be 'zero_mean' or 'minmax'")
            
        self.labels = torch.LongTensor(labels)
        self.obj_ids = obj_ids
    
    def normalize_zero_mean(self, trajectories):
        trajectories_norm = trajectories.copy()
        for i in range(trajectories.shape[0]):
            traj = trajectories[i]
            mean = np.mean(traj)
            std = np.std(traj)
            if std > 0:
                trajectories_norm[i] = (traj - mean) / std
        return torch.FloatTensor(trajectories_norm)
    
    def normalize_minmax(self, trajectories, feature_range=(-1, 1)):
        trajectories_norm = trajectories.copy()
        min_val, max_val = feature_range
        
        for i in range(trajectories.shape[0]):
            traj = trajectories[i]
            traj_min = np.min(traj)
            traj_max = np.max(traj)
            
            if traj_max - traj_min > 0:
                trajectories_norm[i] = (traj - traj_min) / (traj_max - traj_min)
                trajectories_norm[i] = trajectories_norm[i] * (max_val - min_val) + min_val
            else:
                trajectories_norm[i] = np.full_like(traj, (min_val + max_val) / 2)
                
        return torch.FloatTensor(trajectories_norm)
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        if self.obj_ids is not None:
            return self.trajectories[idx], self.labels[idx], self.obj_ids[idx]
        else:
            return self.trajectories[idx], self.labels[idx]


def prepare_data(data_path, seq_length=191, features=['x', 'y', 'xvel', 'yvel']):
    """
    Load and prepare trajectory data from various formats.
    This is a template - you should customize it for your data format.
    """
    # Example implementation for HDF5 files
    df = pd.read_hdf(data_path)
    
    # Group by object ID and create trajectories
    trajectories = []
    labels = []
    obj_ids = []
    

    for obj_id, group in df.groupby('obj_id_unique'):
        # Sort by time
        group = group.sort_values('time stamp')
        
        # Extract features
        traj = group[features].values
        
        # Pad or truncate to seq_length
       # if len(traj) > seq_length:
       #     traj = traj[:seq_length]
       # elif len(traj) < seq_length:
       #     padding = np.zeros((seq_length - len(traj), len(features)))
       #     traj = np.vstack([traj, padding])
        
        trajectories.append(traj)
        
        # Get label (customize based on your data)
        if 'dataset_numeric' in group.columns:
            label = group['dataset_numeric'].iloc[0]
            labels.append(label)
        
        obj_ids.append(obj_id)
    
    trajectories = np.array(trajectories)
    
    # Convert labels to numeric if they're strings
    #from sklearn.preprocessing import LabelEncoder
    #if len(labels) > 0 and isinstance(labels[0], str):
    #    le = LabelEncoder()
    #    labels = le.fit_transform(labels)
    
    return trajectories, np.array(labels), np.array(obj_ids)

