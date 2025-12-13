"""
Diagnostic and visualization functions for trajectory VAE analysis.
"""
import sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import NearestNeighbors
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QPushButton, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
#import ipywidgets as widgets
#from IPython.display import display, clear_output
import torch




"""
Diagnostic plotting functions including beta visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_progress(metrics, save_path=None):
    """Plot training progress with optional beta values"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot total losses
    axes[0, 0].plot(metrics['train_losses'], label='Train')
    axes[0, 0].plot(metrics['val_losses'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot reconstruction losses
    axes[0, 1].plot(metrics.get('recon_losses', []), label='Train Recon')
    axes[0, 1].plot(metrics.get('val_recon_losses', []), label='Val Recon')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot KL losses
    axes[0, 2].plot(metrics['kl_losses'], label='Train KL')
    axes[0, 2].plot(metrics['val_kl_losses'], label='Val KL')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('KL Loss')
    axes[0, 2].set_title('KL Divergence Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot triplet losses
    axes[1, 0].plot(metrics['triplet_losses'], label='Train Triplet')
    axes[1, 0].plot(metrics['val_triplet_losses'], label='Val Triplet')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Triplet Loss')
    axes[1, 0].set_title('Triplet Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot KL vs Reconstruction ratio
    if 'recon_losses' in metrics and 'kl_losses' in metrics:
        kl_recon_ratio = np.array(metrics['kl_losses']) / np.array(metrics['recon_losses'])
        axes[1, 1].plot(kl_recon_ratio)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('KL/Recon Ratio')
        axes[1, 1].set_title('KL to Reconstruction Loss Ratio')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot beta values if available
    if 'beta_values' in metrics:
        axes[1, 2].plot(metrics['beta_values'], 'g-', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Beta Value')
        axes[1, 2].set_title('KL Beta Annealing Schedule')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to: {save_path}")
    
    plt.close(fig)


def plot_latent_space(model, test_loader, labels, reduce='PCA', KDE=True, 
                     n_feats=48, path='latent_space.png', class_labels=None, 
                     train_colors=None, real_colors=None):
    """Plot latent space visualization"""
    
    model.eval()
    latent_vectors = []
    true_labels = []
    obj_ids = []
    scaler = StandardScaler()
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                data, label, obj_id_batch = batch
                obj_ids.extend(obj_id_batch)
            elif len(batch) == 2:
                data, label = batch
            
            mu, logvar = model.encode(data)
            latent_vectors.append(mu.numpy())
            true_labels.extend(label.numpy())
    
    latent_vectors = np.vstack(latent_vectors)
    true_labels = np.array(true_labels)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot first two latent dimensions
    unique_labels = np.unique(true_labels)
    
    if train_colors is None:
        cmap_train = plt.cm.get_cmap('viridis', len(unique_labels))
    else:
        custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", train_colors)
        cmap_train = custom_cmap
    
    scatter1 = ax1.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                          c=true_labels, cmap=cmap_train, alpha=0.7,
                          vmin=min(unique_labels), vmax=max(unique_labels))

    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    ax1.set_title('First Two Latent Dimensions')
    cbar1 = plt.colorbar(scatter1, ax=ax1, ticks=unique_labels)
    cbar1.set_label('Class')
    cbar1.set_ticklabels([class_labels.get(label, f'Class {label}') for label in unique_labels])
    


    # Dimensionality reduction
    if reduce == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=42)
        latent_reduced = reducer.fit_transform(latent_vectors[:, :n_feats])
    elif reduce == 'PCA':
        reducer = PCA(n_components=2)
        latent_reduced = reducer.fit_transform(scaler.fit_transform(latent_vectors[:, :n_feats]))
        print(f'PCA explained variance: {reducer.explained_variance_ratio_}')
    elif reduce == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        latent_reduced = reducer.fit_transform(latent_vectors[:, :n_feats])
    
    # Plot reduced space
    if KDE:
        sns.kdeplot(x=latent_reduced[:, 0], y=latent_reduced[:, 1],
                   hue=true_labels, levels=10, thresh=0.05, fill=True,
                   alpha=0.2, palette=cmap_train, legend=False, ax=ax2)
    else:
        ax2.scatter(latent_reduced[:, 0], latent_reduced[:, 1],
                   c=true_labels, cmap=cmap_train, alpha=0.7,
                   vmin=min(unique_labels), vmax=max(unique_labels))
    
    ax2.set_xlabel(f'{reduce} Component 1')
    ax2.set_ylabel(f'{reduce} Component 2')
    ax2.set_title(f'{reduce} Visualization of Latent Space')
    
    plt.tight_layout()
    
    if path:
        plt.savefig(path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return latent_vectors, true_labels, latent_reduced, obj_ids


def plot_reconstructions(model, test_loader, n_input_feats=4, num_samples=20, save_path=None):
    """Plot original vs reconstructed trajectories"""
    
    model.eval()
    with torch.no_grad():
        data_iter = iter(test_loader)
        try:
            data, labels, obj_ids = next(data_iter)
        except ValueError:
            data, labels = next(data_iter)
            obj_ids = None
        
        # Handle both VAE (returns 3 values) and TripletVAE (returns 4 values)
        model_output = model(data)
        
        # Check if model returns 3 or 4 values
        if isinstance(model_output, tuple):
            if len(model_output) == 4:
                recon_data, mu, logvar, z = model_output
            elif len(model_output) == 3:
                recon_data, mu, logvar = model_output
            else:
                # Handle unexpected output
                recon_data = model_output[0]  # Assume first is reconstruction
        else:
            # Model returns single tensor
            recon_data = model_output
        
        fig, axes = plt.subplots(2, num_samples, figsize=(45, 6))
        
        for i in range(min(num_samples, data.shape[0])):
            # Original trajectory
            orig_traj_full = data[i].numpy().reshape(-1, n_input_feats)
            orig_traj = orig_traj_full[:, :2]  # x, y positions
            
            axes[0, i].plot(orig_traj[:, 0], orig_traj[:, 1], 'b-', linewidth=2)
            axes[0, i].scatter(orig_traj[0, 0], orig_traj[0, 1], color='red', s=50)
            axes[0, i].scatter(orig_traj[-1, 0], orig_traj[-1, 1], color='green', s=50)
            
            title_suffix = f" ({obj_ids[i]})" if obj_ids is not None and i < len(obj_ids) else ""
            axes[0, i].set_title(f'Original{title_suffix}')
            axes[0, i].axis('off')
            
            # Reconstructed trajectory
            recon_traj_full = recon_data[i].numpy().reshape(-1, n_input_feats)
            recon_traj = recon_traj_full[:, :2]
            
            axes[1, i].plot(recon_traj[:, 0], recon_traj[:, 1], 'r--', linewidth=2)
            axes[1, i].scatter(recon_traj[0, 0], recon_traj[0, 1], color='red', s=50)
            axes[1, i].scatter(recon_traj[-1, 0], recon_traj[-1, 1], color='green', s=50)
            axes[1, i].set_title(f'Reconstructed{title_suffix}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

def plot_training_progress(metrics, save_path=None):
    """Plot training progress metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(metrics.get('train_losses', []), label='Train')
    axes[0, 0].plot(metrics.get('val_losses', []), label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Triplet loss
    axes[0, 1].plot(metrics.get('triplet_losses', []), label='Train')
    axes[0, 1].plot(metrics.get('val_triplet_losses', []), label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Triplet Loss')
    axes[0, 1].set_title('Triplet Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL loss
    axes[1, 0].plot(metrics.get('kl_losses', []), label='Train')
    axes[1, 0].plot(metrics.get('val_kl_losses', []), label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate schedule (if available)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_trajectory_from_latent(model, latent_point, n_features=4):
    """Generate a trajectory from a latent point"""
    with torch.no_grad():
        latent_tensor = torch.FloatTensor(latent_point.reshape(1, -1))
        generated = model.decode(latent_tensor).numpy()
    
    trajectory = generated.reshape(-1, n_features)
    return trajectory

###### ---------------------------------------------------------------------- ######
### code for interactive 2D latent space projection for testing smoothness #####


class LatentSpaceExplorer(QMainWindow):
    def __init__(self, model, data_loader, labels=None, n_features=4, seq_length=200):
        super().__init__()
        
        self.model = model
        self.n_features = n_features
        self.seq_length = seq_length
        
        # Extract data
        self.prepare_data(data_loader, labels)
        
        # Initialize UI
        self.init_ui()
        
    def prepare_data(self, data_loader, labels):
        """Prepare data for the explorer"""
        print("Loading data...")
        
        all_data = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) >= 1:
                    data = batch[0]
                    all_data.append(data)
                    if len(batch) >= 2:
                        if isinstance(batch[1], torch.Tensor):
                            all_labels.extend(batch[1].numpy())
                        else:
                            all_labels.extend(batch[1])
        
        all_data = torch.cat(all_data, dim=0)
        
        # Get latent vectors
        latent_vectors = []
        with torch.no_grad():
            batch_size = 128
            for i in range(0, len(all_data), batch_size):
                batch = all_data[i:i+batch_size]
                mu, logvar = self.model.encode(batch)
                latent_vectors.append(mu.numpy())
        
        self.latent_vectors = np.vstack(latent_vectors)
        
        # Handle labels
        if len(all_labels) != len(self.latent_vectors):
            self.labels = np.arange(len(self.latent_vectors))
        else:
            self.labels = np.array(all_labels)
        
        # Reduce to 2D
        self.pca = PCA(n_components=2)
        self.latent_2d = self.pca.fit_transform(self.latent_vectors)
        
        # Build nearest neighbors
        self.knn = NearestNeighbors(n_neighbors=5)
        self.knn.fit(self.latent_2d)
        
        self.selected_point = None
        self.current_trajectory = None
        
        print(f"Data ready: {len(self.latent_vectors)} samples")
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Interactive Latent Space Explorer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Plot area layout
        plot_layout = QHBoxLayout()
        
        # Left plot: Latent space
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.fig_latent = Figure(figsize=(6, 5))
        self.ax_latent = self.fig_latent.add_subplot(111)
        self.canvas_latent = FigureCanvas(self.fig_latent)
        self.toolbar_latent = NavigationToolbar(self.canvas_latent, self)
        
        left_layout.addWidget(self.toolbar_latent)
        left_layout.addWidget(self.canvas_latent)
        
        # Right plot: Trajectory
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        self.fig_trajectory = Figure(figsize=(6, 5))
        self.ax_trajectory = self.fig_trajectory.add_subplot(111)
        self.canvas_trajectory = FigureCanvas(self.fig_trajectory)
        self.toolbar_trajectory = NavigationToolbar(self.canvas_trajectory, self)
        
        right_layout.addWidget(self.toolbar_trajectory)
        right_layout.addWidget(self.canvas_trajectory)
        
        plot_layout.addWidget(left_widget)
        plot_layout.addWidget(right_widget)
        main_layout.addLayout(plot_layout)
        
        # Control panel
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        control_layout = QVBoxLayout(control_frame)
        
        # Coordinate display
        self.coord_label = QLabel("PCA 1: 0.000, PCA 2: 0.000")
        self.coord_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        control_layout.addWidget(self.coord_label)
        
        # Sliders
        slider_layout = QHBoxLayout()
        
        # PCA 1 slider
        pca1_layout = QVBoxLayout()
        pca1_layout.addWidget(QLabel("PCA 1:"))
        self.pca1_slider = QSlider(Qt.Horizontal)
        self.pca1_slider.setMinimum(-100)
        self.pca1_slider.setMaximum(100)
        self.pca1_slider.setValue(0)
        pca1_layout.addWidget(self.pca1_slider)
        slider_layout.addLayout(pca1_layout)
        
        # PCA 2 slider
        pca2_layout = QVBoxLayout()
        pca2_layout.addWidget(QLabel("PCA 2:"))
        self.pca2_slider = QSlider(Qt.Horizontal)
        self.pca2_slider.setMinimum(-100)
        self.pca2_slider.setMaximum(100)
        self.pca2_slider.setValue(0)
        pca2_layout.addWidget(self.pca2_slider)
        slider_layout.addLayout(pca2_layout)
        
        control_layout.addLayout(slider_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Trajectory")
        self.reset_btn = QPushButton("Reset View")
        self.export_btn = QPushButton("Export Plot")
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.export_btn)
        
        control_layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("Ready. Click on a point or use sliders.")
        control_layout.addWidget(self.status_label)
        
        main_layout.addWidget(control_frame)
        
        # Initialize plots
        self.init_plots()
        
        # Connect signals
        self.pca1_slider.valueChanged.connect(self.on_slider_changed)
        self.pca2_slider.valueChanged.connect(self.on_slider_changed)
        self.save_btn.clicked.connect(self.save_trajectory)
        self.reset_btn.clicked.connect(self.reset_view)
        self.export_btn.clicked.connect(self.export_plot)
        
        # Connect canvas click
        self.canvas_latent.mpl_connect('button_press_event', self.on_latent_click)
        
        # Initial update
        self.update_from_sliders()
    
    def init_plots(self):
        """Initialize both plots"""
        # Latent space plot
        self.ax_latent.clear()
        scatter = self.ax_latent.scatter(self.latent_2d[:, 0], self.latent_2d[:, 1],
                                        c=self.labels, cmap='viridis',
                                        alpha=0.6, s=20, picker=True)
        self.ax_latent.set_xlabel('PCA 1')
        self.ax_latent.set_ylabel('PCA 2')
        self.ax_latent.set_title('Latent Space - Click to Generate')
        self.ax_latent.grid(True, alpha=0.3)
        
        # Add colorbar
        self.fig_latent.colorbar(scatter, ax=self.ax_latent, label='Label')
        
        # Trajectory plot
        self.ax_trajectory.clear()
        self.ax_trajectory.set_xlabel('X Position')
        self.ax_trajectory.set_ylabel('Y Position')
        self.ax_trajectory.set_title('Generated Trajectory')
        self.ax_trajectory.grid(True, alpha=0.3)
        self.ax_trajectory.set_aspect('equal', adjustable='datalim')
        
        self.canvas_latent.draw()
        self.canvas_trajectory.draw()
    
    def on_latent_click(self, event):
        """Handle clicks on the latent space plot"""
        if event.inaxes == self.ax_latent:
            # Convert click to PCA coordinates
            pca1_val = event.xdata
            pca2_val = event.ydata
            
            # Convert to slider values (-100 to 100 range)
            pca1_range = self.latent_2d[:, 0].max() - self.latent_2d[:, 0].min()
            pca2_range = self.latent_2d[:, 1].max() - self.latent_2d[:, 1].min()
            
            pca1_slider_val = int(((pca1_val - self.latent_2d[:, 0].min()) / pca1_range * 200) - 100)
            pca2_slider_val = int(((pca2_val - self.latent_2d[:, 1].min()) / pca2_range * 200) - 100)
            
            # Update sliders
            self.pca1_slider.setValue(pca1_slider_val)
            self.pca2_slider.setValue(pca2_slider_val)
            
            # Generate trajectory
            self.generate_trajectory(pca1_val, pca2_val)
    
    def on_slider_changed(self):
        """Handle slider changes"""
        self.update_from_sliders()
    
    def update_from_sliders(self):
        """Update based on current slider values"""
        # Convert slider values to PCA coordinates
        pca1_slider_val = self.pca1_slider.value()
        pca2_slider_val = self.pca2_slider.value()
        
        pca1_range = self.latent_2d[:, 0].max() - self.latent_2d[:, 0].min()
        pca2_range = self.latent_2d[:, 1].max() - self.latent_2d[:, 1].min()
        
        pca1_val = self.latent_2d[:, 0].min() + (pca1_slider_val + 100) / 200 * pca1_range
        pca2_val = self.latent_2d[:, 1].min() + (pca2_slider_val + 100) / 200 * pca2_range
        
        # Update coordinate display
        self.coord_label.setText(f"PCA 1: {pca1_val:.3f}, PCA 2: {pca2_val:.3f}")
        
        # Update selected point marker
        if self.selected_point is not None:
            self.selected_point.remove()
        
        self.selected_point = self.ax_latent.plot(pca1_val, pca2_val, 'rx', 
                                                 markersize=15, markeredgewidth=2)[0]
        self.canvas_latent.draw()
        
        # Generate trajectory
        self.generate_trajectory(pca1_val, pca2_val)
    
    def generate_trajectory(self, pca1_val, pca2_val):
        """Generate and display trajectory"""
        try:
            # Find nearest neighbors
            distances, indices = self.knn.kneighbors([[pca1_val, pca2_val]])
            
            # Weighted average
            weights = 1.0 / (distances[0] + 1e-8)
            weights = weights / weights.sum()
            
            # Create blended latent vector
            blended_latent = np.zeros_like(self.latent_vectors[0])
            for i, weight in enumerate(weights):
                blended_latent += weight * self.latent_vectors[indices[0][i]]
            
            # Generate trajectory
            with torch.no_grad():
                latent_tensor = torch.FloatTensor(blended_latent.reshape(1, -1))
                generated = self.model.decode(latent_tensor).numpy()
            
            # Reshape
            trajectory = generated.reshape(self.seq_length, self.n_features)
            self.current_trajectory = trajectory
            
            x_pos = trajectory[:, 0]
            y_pos = trajectory[:, 1]
            
            # Update trajectory plot
            self.ax_trajectory.clear()
            self.ax_trajectory.plot(x_pos, y_pos, 'b-', linewidth=2, label='Trajectory')
            self.ax_trajectory.plot(x_pos[0], y_pos[0], 'go', markersize=10, label='Start')
            self.ax_trajectory.plot(x_pos[-1], y_pos[-1], 'ro', markersize=10, label='End')
            
            # Add velocity arrows
            if self.n_features >= 4:
                skip = max(1, self.seq_length // 20)
                for i in range(0, self.seq_length, skip):
                    self.ax_trajectory.arrow(x_pos[i], y_pos[i], 
                                           trajectory[i, 2] * 0.1, trajectory[i, 3] * 0.1,
                                           head_width=0.05, head_length=0.1,
                                           fc='orange', ec='orange', alpha=0.6)
            
            self.ax_trajectory.set_xlabel('X Position')
            self.ax_trajectory.set_ylabel('Y Position')
            self.ax_trajectory.set_title(f'Generated Trajectory\nPCA: ({pca1_val:.2f}, {pca2_val:.2f})')
            self.ax_trajectory.grid(True, alpha=0.3)
            self.ax_trajectory.legend()
            self.ax_trajectory.set_aspect('equal', adjustable='datalim')
            
            self.status_label.setText(f"Generated from {len(indices[0])} nearest points")
            
            self.canvas_trajectory.draw()
            
        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            print(f"Error: {e}")
    
    def save_trajectory(self):
        """Save current trajectory to file"""
        if self.current_trajectory is not None:
            pca1_val = float(self.coord_label.text().split(':')[1].split(',')[0].strip())
            filename = f"trajectory_pca{pca1_val:.2f}.npy"
            np.save(filename, self.current_trajectory)
            self.status_label.setText(f"Saved to {filename}")
    
    def reset_view(self):
        """Reset to default view"""
        self.pca1_slider.setValue(0)
        self.pca2_slider.setValue(0)
        self.update_from_sliders()
    
    def export_plot(self):
        """Export current plot to file"""
        # Save both figures
        self.fig_latent.savefig('latent_space.png', dpi=150, bbox_inches='tight')
        self.fig_trajectory.savefig('trajectory.png', dpi=150, bbox_inches='tight')
        self.status_label.setText("Plots saved as latent_space.png and trajectory.png")


# Function to launch the PyQt5 GUI
def launch_pyqt_gui(model, data_loader, labels=None, n_features=4, seq_length=200):
    """
    Launch the PyQt5 GUI for latent space exploration.
    """
    app = QApplication(sys.argv)
    
    # If in Jupyter, we need to handle this differently
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except:
        in_notebook = False
    
    if in_notebook:
        print("Note: PyQt GUI launched in separate process.")
        print("Close the window when done exploring.")
    
    explorer = LatentSpaceExplorer(model, data_loader, labels, n_features, seq_length)
    explorer.show()
    
    if not in_notebook:
        sys.exit(app.exec_())
    else:
        # In notebook, we can't block, so just return
        app.exec_()
    
    return explorer