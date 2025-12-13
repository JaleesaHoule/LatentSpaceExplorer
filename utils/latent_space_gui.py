# latent_space_gui.py
import io
import sys
import os
import time
import traceback
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QSlider, QPushButton, QFrame,
                            QComboBox, QGroupBox, QRadioButton, QButtonGroup,
                            QGridLayout, QColorDialog, QDialog, QListWidget,
                            QListWidgetItem, QDialogButtonBox, QSplitter, QSizePolicy,
                             QTextEdit, QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QColor


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import multiprocessing as mp
from sklearn.decomposition import PCA





# Set start method to 'spawn' to avoid fork issues
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)
print(f"Multiprocessing start method: {mp.get_start_method()}")

class TextCapture:
    """Capture stdout and stderr to display in GUI."""
    def __init__(self, text_widget, log_method=None):
        self.text_widget = text_widget
        self.log_method = log_method
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        self.buffer = io.StringIO()
    
    def write(self, text):
        # Write to original stdout as well
        self.old_stdout.write(text)
        
        # Write to buffer
        self.buffer.write(text)
        
        # If there's text to display
        if text.strip():
            # Use log method if provided, otherwise append directly
            if self.log_method:
                self.log_method(text.rstrip())
            else:
                self.text_widget.append(text.rstrip())
                # Auto-scroll
                scrollbar = self.text_widget.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
    
    def flush(self):
        self.buffer.flush()
        self.old_stdout.flush()
    
    def start_capture(self):
        sys.stdout = self
        sys.stderr = self
    
    def stop_capture(self):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

class GUIManager:
    """Manager class for GUI processes with proper cleanup."""
    
    def __init__(self):
        self.process = None
    
    def launch_gui(self, model, train_loader, test_loader=None,
                   train_labels=None, test_labels=None,
                   n_features=4, seq_length=200, batch_size=64,
                   initial_mode='generate_trajectories',
                   custom_labels=None, custom_colors=None):
        """
        Launch GUI with proper cleanup handling.
        
        Returns:
            The process object if successful, None otherwise.
        """
        print("\n" + "="*60)
        print("Launching Enhanced Latent Space Explorer")
        print("="*60)
        
        try:
            # Prepare data
            gui_data = prepare_data_for_gui(
                model, train_loader, test_loader,
                train_labels, test_labels, 
                n_features, seq_length, batch_size,
                custom_labels, custom_colors
            )
            
            if gui_data is None:
                print("Failed to prepare data for GUI")
                return None
            
            # Create and start process
            self.process = mp.Process(
                target=self._run_gui_safely,
                args=(model, gui_data, initial_mode),
                daemon=False,  # Non-daemon so we can manage it
                name="LatentSpaceExplorerGUI"
            )
            
            print("Starting GUI process...")
            self.process.start()
            
            print(f"✓ GUI process started with PID: {self.process.pid}")
            print("\nInstructions:")
            print("1. The GUI will open in a separate window")
            print("2. Close the GUI window when done")
            print("3. Use stop_gui() to force stop if needed")
            print("="*60 + "\n")
            
            return self.process
            
        except Exception as e:
            print(f"✗ Error launching GUI: {e}")
            traceback.print_exc()
            self.cleanup()
            return None
    
    def _run_gui_safely(self, model, gui_data, initial_mode):
        """
        Run GUI in a separate process with error handling and cleanup.
        """
        try:
            # Import here to avoid issues with multiprocessing
            import sys
            import os
            
            # Set environment variable for macOS
            if sys.platform == 'darwin':
                os.environ['QT_MAC_WANTS_LAYER'] = '1'
            
            # Create application
            app = QApplication(sys.argv)
            app.setApplicationName("Latent Space Explorer")
            
            # Create and show window
            explorer = EnhancedLatentSpaceExplorer(
                model=model,
                gui_data=gui_data,
                initial_mode=initial_mode
            )
            
            # Ensure proper cleanup on close
            explorer.setAttribute(Qt.WA_DeleteOnClose)
            
            # Show and raise window
            explorer.show()
            explorer.raise_()
            explorer.activateWindow()
            
            # Bring to front
            QTimer.singleShot(100, explorer.raise_)
            QTimer.singleShot(200, explorer.activateWindow)
            
            print("GUI window created and shown")
            
            # Run event loop
            ret_code = app.exec_()
            
            # Graceful exit
            sys.exit(0)
            
        except Exception as e:
            print(f"✗ GUI process error: {e}")
            traceback.print_exc()
            
            # Force exit with error
            sys.exit(1)
    
    def stop_gui(self):
        """Stop the GUI process gracefully."""
        if self.process and self.process.is_alive():
            print(f"Stopping GUI process {self.process.pid}...")
            
            try:
                # Try to terminate gracefully
                self.process.terminate()
                self.process.join(timeout=2)
                
                if self.process.is_alive():
                    # Force kill if still alive
                    print("  Process still alive, forcing kill...")
                    self.process.kill()
                    self.process.join(timeout=1)
                
                print("✓ GUI stopped")
                
            except Exception as e:
                print(f"✗ Error stopping process: {e}")
    
    def cleanup(self):
        """Clean up all resources."""
        self.stop_gui()
    
    def __del__(self):
        """Destructor for cleanup."""
        self.cleanup()
    
    def is_running(self):
        """Check if GUI is running."""
        return self.process is not None and self.process.is_alive()


# Context manager for GUI sessions
def gui_session(model, train_loader, **kwargs):
    """
    Context manager for GUI sessions.
    
    Example:
        with gui_session(model, train_loader, initial_mode='generate_trajectories') as gui:
            # GUI is running...
            pass  # GUI auto-cleans up when block exits
    """
    manager = GUIManager()
    process = None
    
    try:
        process = manager.launch_gui(model, train_loader, **kwargs)
        if process:
            print("GUI session started. Close the window when done.")
            yield process
        else:
            yield None
    finally:
        if manager.is_running():
            print("\nCleaning up GUI session...")
            manager.cleanup()
            time.sleep(0.5)  # Brief pause for cleanup


def launch_enhanced_gui(model, train_loader, test_loader=None,
                       train_labels=None, test_labels=None,
                       n_features=4, seq_length=200, batch_size=64,
                       initial_mode='generate_trajectories',
                       custom_labels=None, custom_colors=None):
    """
    Convenience function to launch GUI using the manager.
    
    Returns:
        GUIManager instance for controlling the GUI.
    """
    manager = GUIManager()
    process = manager.launch_gui(
        model, train_loader, test_loader,
        train_labels, test_labels,
        n_features, seq_length, batch_size,
        initial_mode,
        custom_labels, custom_colors
    )
    
    if process:
        return manager
    else:
        return None


def prepare_data_for_gui(model, train_loader, test_loader=None,
                        train_labels=None, test_labels=None, 
                        n_features=4, seq_length=200,batch_size=64,
                        custom_labels=None, custom_colors=None):
    """
    Prepare data for GUI in main process.
    """
    print("Preparing data for GUI...")
    
    def extract_data(data_loader, labels, ):
        all_data = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                try:
                    data = batch[0]  # Shape: (batch_size, 764)
                    all_data.append(data)
                    # Extract labels from batch[1]
                    batch_labels = batch[1].detach().numpy()
                    all_labels.extend(batch_labels)
                except Exception as e:
                    raise RuntimeError(f"Error extracting data from batch in data_loader: {e}")
        
        if not all_data:
            return None, None, None
        
        # Concatenate all data
        all_data = torch.cat(all_data, dim=0)
        
        # Get latent vectors
        latent_vectors = []
        with torch.no_grad():
            for i in range(0, len(all_data), batch_size):
                batch = all_data[i:i+batch_size]
                try:
                    mu, logvar = model.encode(batch)
                    latent_vectors.append(mu.detach().numpy())

                except Exception as e:
                    raise RuntimeError(f"Error extracting latent vectors: {e}")
        
        if not latent_vectors:
            raise RuntimeError("No latent vectors extracted!")
        
        latent_vectors = np.vstack(latent_vectors)
        all_labels = np.array(all_labels)
        
        print(f"Data shape: {all_data.shape}")
        print(f"Latent vectors shape: {latent_vectors.shape}")
        print(f"Labels shape: {all_labels.shape}")
        
        return all_data.detach().numpy(), latent_vectors, all_labels
    
    # Extract training data
    print("Extracting training data...")
    train_data, train_latent, train_labels_array = extract_data(train_loader, train_labels)
    
    if train_data is None:
        raise RuntimeError("No training data found!")
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Training latent shape: {train_latent.shape}")
    print(f"Training labels shape: {train_labels_array.shape}")
    
    # Extract test data if provided
    if test_loader is not None:
        print("Extracting test data...")
        test_data, test_latent, test_labels_array = extract_data(test_loader, test_labels)
        if test_data is not None:
            print(f"Test data shape: {test_data.shape}")
            print(f"Test latent shape: {test_latent.shape}")
    else:
        test_data, test_latent, test_labels_array = None, None, None
    
    # Compute PCA on training data
    print("Computing PCA...")
    try:
        pca = PCA(n_components=2)
        train_latent_2d = pca.fit_transform(train_latent)
        
        print(f"PCA components shape: {pca.components_.shape}")
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")
        
    except Exception as e:
        raise RuntimeError(f"Error computing PCA: {e}")
    
    # Transform test data if available
    if test_latent is not None:
        test_latent_2d = pca.transform(test_latent)
    else:
        test_latent_2d = None
    
    # Get unique classes for training
    train_classes = np.unique(train_labels_array)
    print(f"Training classes: {train_classes}")
    
    # Compute KDEs for each class
    kde_data = {}
    
    print("Computing KDEs...")
    for cls in train_classes:
        mask = train_labels_array == cls
        if np.sum(mask) > 10:  # Need enough points for KDE
            try:
                class_points = train_latent_2d[mask]
                if len(class_points) > 1000:
                    indices = np.random.choice(len(class_points), 1000, replace=False)
                    class_points = class_points[indices]
                
                kde = gaussian_kde(class_points.T)
                kde_data[cls] = (kde, class_points)
                print(f"  Class {cls}: KDE computed with {len(class_points)} points")
            except Exception as e:
                print(f"  Could not compute KDE for class {cls}: {e}")
                kde_data[cls] = (None, None)
        else:
            print(f"  Class {cls}: Not enough points ({np.sum(mask)}) for KDE")
            kde_data[cls] = (None, None)
    
    # Calculate model input dimension
    if len(train_data.shape) == 2:
        model_input_dim = train_data.shape[1]
        print(f"Model input dimension (flattened): {model_input_dim}")
    else:
        model_input_dim = seq_length * n_features
        print(f"Using calculated model input dimension: {model_input_dim}")
    
    # Prepare data dictionary
    gui_data = {
        'train_data': train_data,
        'train_latent': train_latent,
        'train_latent_2d': train_latent_2d,
        'train_labels': train_labels_array,
        'train_classes': train_classes,
        'test_data': test_data,
        'test_latent': test_latent,
        'test_latent_2d': test_latent_2d,
        'test_labels': test_labels_array,
        'kde_data': kde_data,
        'pca': pca,
        'n_features': n_features,
        'seq_length': seq_length,
        'custom_labels': custom_labels if custom_labels else {},
        'custom_colors': custom_colors if custom_colors else {},
        'model_input_dim': model_input_dim
    }
    
    print(f"\n✓ GUI data prepared successfully!")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data) if test_data is not None else 0}")
    print(f"  Number of classes: {len(train_classes)}")
    print(f"  PCA dimensions: {train_latent_2d.shape}")
    
    return gui_data


class ColorLabelDialog(QDialog):
    """Dialog for customizing class colors and labels"""
    def __init__(self, class_indices, current_labels=None, current_colors=None, parent=None):
        super().__init__(parent)
        self.class_indices = class_indices
        self.setWindowTitle("Customize Class Labels & Colors")
        self.setGeometry(300, 300, 500, 400)
        
        layout = QVBoxLayout(self)
        
        # List widget for classes
        self.list_widget = QListWidget()
        layout.addWidget(QLabel("Classes:"))
        layout.addWidget(self.list_widget)
        
        # Initialize with default values
        self.labels = {}
        self.colors = {}
        
        for idx in class_indices:
            if current_labels and idx in current_labels:
                label = current_labels[idx]
            else:
                label = f"Class {idx}"
            
            if current_colors and idx in current_colors:
                color = current_colors[idx]
            else:
                # Default color from tab20 colormap
                cmap = plt.cm.tab20
                color_idx = idx % 20
                color = cmap(color_idx)
                color = (color[0], color[1], color[2])  # RGB tuple
            
            self.labels[idx] = label
            self.colors[idx] = color
            
            # Create list item
            item = QListWidgetItem(f"{idx}: {label}")
            item.setData(Qt.UserRole, idx)
            
            # Set background color
            qcolor = QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255))
            item.setBackground(qcolor)
            
            # Make text readable
            brightness = (color[0]*299 + color[1]*587 + color[2]*114) / 1000
            text_color = QColor(255, 255, 255) if brightness < 0.5 else QColor(0, 0, 0)
            item.setForeground(text_color)
            
            self.list_widget.addItem(item)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        edit_btn = QPushButton("Edit Selected")
        edit_btn.clicked.connect(self.edit_selected)
        button_layout.addWidget(edit_btn)
        
        layout.addLayout(button_layout)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def edit_selected(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        class_idx = item.data(Qt.UserRole)
        current_label = self.labels[class_idx]
        current_color = self.colors[class_idx]
        
        # Create edit dialog
        edit_dialog = QDialog(self)
        edit_dialog.setWindowTitle(f"Edit Class {class_idx}")
        edit_dialog.setGeometry(350, 350, 300, 200)
        
        layout = QVBoxLayout(edit_dialog)
        
        # Label edit
        layout.addWidget(QLabel("Label:"))
        label_edit = QComboBox()
        label_edit.setEditable(True)
        label_edit.addItems([
            "Class 0", "Class 1", "Class 2", "Class 3", "Class 4",
            "Class 5", "Class 6", "Class 7", "Class 8", "Class 9",
            "Circling", "Line", "Sine", "Brownian", "Random Walk",
            "Levy Flight", "Run and Tumble", "Casting", "Random Turning", "Other"
        ])
        label_edit.setCurrentText(current_label)
        layout.addWidget(label_edit)
        
        # Color selection
        layout.addWidget(QLabel("Color:"))
        color_btn = QPushButton("Choose Color")
        color_btn.clicked.connect(lambda: self.choose_color(color_btn, current_color))
        layout.addWidget(color_btn)
        
        # Preview
        preview_label = QLabel("Preview")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setMinimumHeight(40)
        qcolor = QColor(int(current_color[0]*255), int(current_color[1]*255), int(current_color[2]*255))
        preview_label.setStyleSheet(f"background-color: {qcolor.name()}; color: white;")
        layout.addWidget(preview_label)
        
        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        
        def accept_edit():
            new_label = label_edit.currentText()
            new_color = self.colors[class_idx]
            
            self.labels[class_idx] = new_label
            
            # Update list item
            item.setText(f"{class_idx}: {new_label}")
            qcolor = QColor(int(new_color[0]*255), int(new_color[1]*255), int(new_color[2]*255))
            item.setBackground(qcolor)
            
            brightness = (new_color[0]*299 + new_color[1]*587 + new_color[2]*114) / 1000
            text_color = QColor(255, 255, 255) if brightness < 0.5 else QColor(0, 0, 0)
            item.setForeground(text_color)
            
            edit_dialog.accept()
        
        buttons.accepted.connect(accept_edit)
        buttons.rejected.connect(edit_dialog.reject)
        layout.addWidget(buttons)
        
        # Store references
        color_btn.dialog = edit_dialog
        color_btn.preview_label = preview_label
        color_btn.class_idx = class_idx
        color_btn.parent_dialog = self
        
        edit_dialog.exec_()
    
    def choose_color(self, button, current_color):
        qcolor = QColor(int(current_color[0]*255), int(current_color[1]*255), int(current_color[2]*255))
        color = QColorDialog.getColor(qcolor, button.dialog)
        
        if color.isValid():
            new_color = (color.red()/255.0, color.green()/255.0, color.blue()/255.0)
            button.parent_dialog.colors[button.class_idx] = new_color
            button.preview_label.setStyleSheet(f"background-color: {color.name()}; color: white;")


class EnhancedLatentSpaceExplorer(QMainWindow):
    def __init__(self, model, gui_data, initial_mode='generate_trajectories'):
        super().__init__()
        
        self.model = model  # Store the model directly
        self.gui_data = gui_data
        self.mode = initial_mode
        
        # Extract data from gui_data
        self.train_data = gui_data['train_data']
        self.train_latent = gui_data['train_latent']
        self.train_latent_2d = gui_data['train_latent_2d']
        self.train_labels = gui_data['train_labels']
        self.train_classes = gui_data['train_classes']
        
        self.test_data = gui_data['test_data']
        self.test_latent = gui_data['test_latent']
        self.test_latent_2d = gui_data['test_latent_2d']
        self.test_labels = gui_data['test_labels']
        
        self.kde_data = gui_data['kde_data']
        self.pca = gui_data['pca']
        self.n_features = gui_data['n_features']
        self.seq_length = gui_data['seq_length']
        self.custom_labels = gui_data['custom_labels']
        self.custom_colors = gui_data['custom_colors']
        self.explained_variance = gui_data['pca'].explained_variance_ratio_.sum()
        
        # Debug model architecture
        self._debug_model_architecture()
        
        # Reshape data
        self._reshape_data()
        
        # Current state
        self.selected_point = None
        self.current_trajectory = None
        self.original_trajectory = None
        self.is_test_point = False
        
        # Initialize UI
        self.init_ui()
        
        # Apply initial mode
        self.set_mode(initial_mode)
    
    def _debug_model_architecture(self):
        """Debug function to check model architecture."""
        print(f"\n=== DEBUG Model Architecture ===")
        print(f"Model type: {type(self.model)}")
        
        # Check if model has decoder
        if hasattr(self.model, 'decoder'):
            print("✓ Model has 'decoder' attribute")
            decoder = self.model.decoder
            if isinstance(decoder, torch.nn.Sequential):
                print(f"Decoder is Sequential with {len(decoder)} layers:")
                for i, layer in enumerate(decoder):
                    print(f"  Layer {i}: {layer}")
                    if hasattr(layer, 'in_features'):
                        print(f"    in_features: {layer.in_features}, out_features: {layer.out_features}")
        else:
            print("✗ Model does not have 'decoder' attribute")
            print("Will try to use model directly...")
    
    def _reshape_data(self):
        """Reshape flattened data back to trajectories."""
        print(f"Reshaping data: {self.train_data.shape} -> (-1, {self.seq_length}, {self.n_features})")
        
        if len(self.train_data.shape) == 2:
            total_elements = self.train_data.shape[0] * self.train_data.shape[1]
            expected_elements = (self.train_data.shape[0] * self.seq_length * self.n_features)
            
            if total_elements == expected_elements:
                self.train_data = self.train_data.reshape(-1, self.seq_length, self.n_features)
            else:
                print(f"Warning: Data size mismatch. Flattening first...")
                self.train_data = self.train_data.flatten().reshape(-1, self.seq_length, self.n_features)
        
        if self.test_data is not None and len(self.test_data.shape) == 2:
            total_elements_test = self.test_data.shape[0] * self.test_data.shape[1]
            expected_test = (self.test_data.shape[0] * self.seq_length * self.n_features)
            
            if total_elements_test == expected_test:
                self.test_data = self.test_data.reshape(-1, self.seq_length, self.n_features)
            else:
                self.test_data = self.test_data.flatten().reshape(-1, self.seq_length, self.n_features)



    def init_ui(self):
        self.setWindowTitle('Enhanced Latent Space Explorer')
        self.setGeometry(100, 100, 1600, 1000)
    
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
    
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)  # Reduced spacing

        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
    
        # ========== MODIFIED SECTION: Use QSplitter ==========
        # Create vertical splitter for plots and console
        vertical_splitter = QSplitter(Qt.Vertical)
    
        # Create horizontal splitter for left/right plots
        plot_splitter = QSplitter(Qt.Horizontal)
    
        # Left plot
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        self.fig_latent = Figure(figsize=(7, 5))
        self.fig_latent.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
        self.ax_latent = self.fig_latent.add_subplot(111)
        self.canvas_latent = FigureCanvas(self.fig_latent)
        self.toolbar_latent = NavigationToolbar(self.canvas_latent, self)
    
        left_layout.addWidget(self.toolbar_latent)
        left_layout.addWidget(self.canvas_latent)
    
        # Right plot
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)

        self.fig_trajectory = Figure(figsize=(7, 5))
        self.fig_latent.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
        self.ax_trajectory = self.fig_trajectory.add_subplot(111)
        self.canvas_trajectory = FigureCanvas(self.fig_trajectory)
        self.toolbar_trajectory = NavigationToolbar(self.canvas_trajectory, self)
    
        right_layout.addWidget(self.toolbar_trajectory)
        right_layout.addWidget(self.canvas_trajectory)
    
        # Add plot widgets to horizontal splitter
        plot_splitter.addWidget(left_widget)
        plot_splitter.addWidget(right_widget)
    
        # Set initial sizes for plots (equal width)
        plot_splitter.setSizes([800, 800])
    
        # Console output
        console_group = QGroupBox("Console Output")
        console_layout = QVBoxLayout()
    
        # Create text widget for console output
        self.console_text = QTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setStyleSheet(""" QTextEdit {
            background-color: #ffffff;
            color: #1e1e1e;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            border: 1px solid #555;} """)
    
        # Add control buttons
        button_layout = QHBoxLayout()
    
        clear_btn = QPushButton("Clear Console")
        clear_btn.clicked.connect(self.clear_console)
        button_layout.addWidget(clear_btn)
    
        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self.copy_console_to_clipboard)
        button_layout.addWidget(copy_btn)
    
        save_btn = QPushButton("Save to File")
        save_btn.clicked.connect(self.save_console_to_file)
        button_layout.addWidget(save_btn)
    
        button_layout.addStretch()
    
        console_layout.addLayout(button_layout)
        console_layout.addWidget(self.console_text)
        console_group.setLayout(console_layout)
    
        # Add widgets to vertical splitter
        vertical_splitter.addWidget(plot_splitter)
        vertical_splitter.addWidget(console_group)
    
        # Set initial sizes (plots get 70%, console gets 30%)
        vertical_splitter.setSizes([700, 300])
    
        # Add vertical splitter to main layout
        main_layout.addWidget(vertical_splitter, stretch=1)
        # ========== END MODIFIED SECTION ==========
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage(f"Ready - PCA Explained Variance: {self.explained_variance:.3f}")
    
        # Connect canvas click
        self.canvas_latent.mpl_connect('button_press_event', self.on_latent_click)
    
        # Start capturing output
        self.start_output_capture()





    def start_output_capture(self):
        """Start capturing stdout and stderr."""
        self.text_capture = TextCapture(self.console_text, self.log_message)
        self.text_capture.start_capture()
        self.log_message("Console output capture started...")
    
    def log_message(self, message):
        """Add a message to the console text box."""
        self.console_text.append(str(message))
        # Auto-scroll to bottom
        scrollbar = self.console_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_console(self):
        """Clear the console text box."""
        self.console_text.clear()
        self.log_message("Console cleared.")
    
    def copy_console_to_clipboard(self):
        """Copy console content to clipboard."""
        text = self.console_text.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.log_message("Console output copied to clipboard.")
            self.status_bar.showMessage("Console output copied to clipboard", 3000)
    
    def save_console_to_file(self):
        """Save console content to a text file."""
        from PyQt5.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Console Output", "", "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.console_text.toPlainText())
                self.log_message(f"Console output saved to {filename}")
                self.status_bar.showMessage(f"Output saved to {filename}", 3000)
            except Exception as e:
                self.log_message(f"Error saving file: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop capturing output
        if self.text_capture:
            self.text_capture.stop_capture()
        
        # Call parent close event
        super().closeEvent(event)



    def create_control_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        layout = QGridLayout(panel)
        
        # Mode selection
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout()
        
        self.mode_buttons = QButtonGroup()
        
        self.mode_generate = QRadioButton("Generate Trajectories (Direct PCA Inverse)")
        self.mode_reconstruct = QRadioButton("Reconstruct Training Trajectories")
        self.mode_project = QRadioButton("Project Unseen Test Trajectories")
        
        self.mode_buttons.addButton(self.mode_generate)
        self.mode_buttons.addButton(self.mode_reconstruct)
        self.mode_buttons.addButton(self.mode_project)
        
        mode_layout.addWidget(self.mode_generate)
        mode_layout.addWidget(self.mode_reconstruct)
        mode_layout.addWidget(self.mode_project)
        
        mode_group.setLayout(mode_layout)


        # Create a mapping between buttons and modes
        self.mode_mapping = {
            self.mode_generate: 'generate_trajectories',
            self.mode_reconstruct: 'reconstruct_trajectories', 
            self.mode_project: 'project_new_trajectories'
            }
    
        # Connect using QButtonGroup signal
        self.mode_buttons.buttonClicked.connect(self.on_mode_button_clicked)
    
        layout.addWidget(mode_group, 0, 0, 2, 1)
        
        # PCA Variance display
        variance_label = QLabel(f"PCA Explained Variance: {self.explained_variance:.3f}")
        variance_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        layout.addWidget(variance_label, 0, 1)
        
        # Customization button
        custom_btn = QPushButton("Customize Labels & Colors")
        custom_btn.clicked.connect(self.customize_labels_colors)
        layout.addWidget(custom_btn, 1, 1)
        
        # Coordinate display
        self.coord_label = QLabel("Coordinates: ")
        self.coord_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.coord_label, 2, 0, 1, 2)
        
        # Sliders for manual exploration
        slider_group = QGroupBox("Manual Exploration (Generate Mode Only)")
        slider_group.setMaximumHeight(110) 
        slider_layout = QVBoxLayout()
        
        # PCA 1 slider
        pca1_layout = QHBoxLayout()
        pca1_layout.setSpacing(10)
        pca1_layout.addWidget(QLabel("PCA 1:"))
        self.pca1_slider = QSlider(Qt.Horizontal)
        self.pca1_slider.setMinimum(-100)
        self.pca1_slider.setMaximum(100)
        self.pca1_slider.setValue(0)
        self.pca1_slider.setEnabled(False)
        pca1_layout.addWidget(self.pca1_slider)
        slider_layout.addLayout(pca1_layout)
        
        # PCA 2 slider
        pca2_layout = QHBoxLayout()
        pca2_layout.setSpacing(10)
        pca2_layout.addWidget(QLabel("PCA 2:"))
        self.pca2_slider = QSlider(Qt.Horizontal)
        self.pca2_slider.setMinimum(-100)
        self.pca2_slider.setMaximum(100)
        self.pca2_slider.setValue(0)
        self.pca2_slider.setEnabled(False)
        pca2_layout.addWidget(self.pca2_slider)
        slider_layout.addLayout(pca2_layout)
        
        slider_group.setLayout(slider_layout)
        layout.addWidget(slider_group, 3, 0, 1, 2)
        
        # Connect sliders
        self.pca1_slider.valueChanged.connect(self.on_slider_changed)
        self.pca2_slider.valueChanged.connect(self.on_slider_changed)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Trajectory")
        self.reset_btn = QPushButton("Reset View")
        self.export_btn = QPushButton("Export Plot")
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout, 4, 0, 1, 2)
        
        # Connect buttons
        self.save_btn.clicked.connect(self.save_trajectory)
        self.reset_btn.clicked.connect(self.reset_view)
        self.export_btn.clicked.connect(self.export_plot)
        
        return panel

    def on_mode_button_clicked(self, button):
        """Handle mode button clicks."""
        mode = self.mode_mapping.get(button)
        if mode:
            self.on_mode_changed(mode)

    def on_mode_changed(self, mode):
        self.mode = mode
        
        if mode == 'generate_trajectories':
            self.pca1_slider.setEnabled(True)
            self.pca2_slider.setEnabled(True)
            self.status_bar.showMessage(f"Mode: Generate - Click anywhere - PCA Explained Variance: {self.explained_variance:.3f}")
        else:
            self.pca1_slider.setEnabled(False)
            self.pca2_slider.setEnabled(False)
            if mode == 'reconstruct_trajectories':
                self.status_bar.showMessage(f"Mode: Reconstruct - Click on training points - PCA Explained Variance: {self.explained_variance:.3f}")
            else:
                self.status_bar.showMessage(f"Mode: Project - Click on test points - PCA Explained Variance: {self.explained_variance:.3f}")
        
        self.redraw_latent_space()
    
    def customize_labels_colors(self):
        dialog = ColorLabelDialog(self.train_classes, self.custom_labels, self.custom_colors, self)
        if dialog.exec_() == QDialog.Accepted:
            self.custom_labels = dialog.labels
            self.custom_colors = dialog.colors
            self.redraw_latent_space()
    
    def get_class_label(self, class_idx):
        return self.custom_labels.get(class_idx, f"Class {class_idx}")
    
    def get_class_color(self, class_idx):
        if class_idx in self.custom_colors:
            return self.custom_colors[class_idx]
        else:
            cmap = plt.cm.tab20
            color_idx = class_idx % 20
            return cmap(color_idx)
    
    def redraw_latent_space(self):
        self.ax_latent.clear()
        
        if self.mode == 'generate_trajectories':
            # Draw KDE contours for generate mode
            if self.kde_data:
                self.draw_kde_contours()
            self.ax_latent.set_title('2D PCA of Latent Space - Click Anywhere to Generate from PCA Inverse')
            
        elif self.mode == 'reconstruct_trajectories':
            # Draw scatter points for reconstruct mode
            if self.kde_data:
                self.draw_kde_contours()
            self.draw_scatter_points(self.train_latent_2d, self.train_labels, alpha=0.7)
            self.ax_latent.set_title('2D PCA of Latent Space - Click Training Points to Reconstruct')
            
        elif self.mode == 'project_new_trajectories':
            # Draw KDE contours for project mode
            if self.kde_data:
                self.draw_kde_contours(alpha=0.3)
            
            # Draw test points if available
            if self.test_latent_2d is not None:
                self.ax_latent.scatter(self.test_latent_2d[:, 0], self.test_latent_2d[:, 1],
                                      c='black', alpha=0.6, s=30, edgecolors='white',
                                      linewidth=0.5, label='Test Data')
            
            self.ax_latent.set_title('2D PCA of Latent Space - Click Test Points to Reconstruct')
        
        # Add PCA variance to plot
        #self.ax_latent.text(0.02, 0.98, f'PCA Variance: {self.explained_variance:.3f}',
        #                   transform=self.ax_latent.transAxes,
        #                   fontsize=10, verticalalignment='top',
        #                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        self.ax_latent.set_xlabel('PCA 1')
        self.ax_latent.set_ylabel('PCA 2')
        self.ax_latent.grid(True, alpha=0.3)
        '''
        # Add legend for scatter points in reconstruct/project modes
        if self.mode in ['reconstruct_trajectories', 'project_new_trajectories']:
            # Create a custom legend for classes
            if self.mode == 'reconstruct_trajectories':
                unique_labels = np.unique(self.train_labels)
                print(unique_labels)
                for label in unique_labels:
                    color = self.get_class_color(label)
                    class_label = self.get_class_label(label)
                    self.ax_latent.scatter([], [], c=[color], alpha=0.7, s=30,
                                          edgecolors='white', linewidth=0.5,
                                          label=class_label)
            
            self.ax_latent.legend(bbox_to_anchor=(1, 1), fontsize=9)
        '''

        # Add legend - but ONLY if we have labels
        # The scatter points already have labels from draw_scatter_points
        handles, labels = self.ax_latent.get_legend_handles_labels()
    
        # Remove duplicate labels while preserving order
        unique_labels = []
        unique_handles = []
        seen = set()
    
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)
                unique_handles.append(handle)
    
        if unique_handles:
            self.ax_latent.legend(unique_handles, unique_labels, bbox_to_anchor=(1, 1), fontsize=9)
    
        self.fig_latent.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas_latent.draw()


    def draw_kde_contours(self, alpha=0.5):
        """Draw KDE contours with visible outlines."""
        if not self.kde_data:
            return
    
        # Get plot bounds
        x_min, x_max = self.train_latent_2d[:, 0].min(), self.train_latent_2d[:, 0].max()
        y_min, y_max = self.train_latent_2d[:, 1].min(), self.train_latent_2d[:, 1].max()
    
        x_padding = (x_max - x_min) * 0.1  # Increased padding
        y_padding = (y_max - y_min) * 0.1
        x_grid = np.linspace(x_min - x_padding, x_max + x_padding, 150)
        y_grid = np.linspace(y_min - y_padding, y_max + y_padding, 150)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[xx.ravel(), yy.ravel()]
     
        for cls in self.train_classes:
            if cls in self.kde_data and self.kde_data[cls][0] is not None:
                try:
                    kde, class_points = self.kde_data[cls]
                    if kde is None:
                        continue
                
                    kde_vals = kde(grid_points.T).reshape(xx.shape)
                
                    #Apply threshold to remove near-zero values
                    threshold = kde_vals.max() * 0.05  # 5% of max density
                    kde_vals_masked = np.where(kde_vals < threshold, np.nan, kde_vals)
                
                    # Optional: Use log scale for better visualization
                    # kde_vals_masked = np.log(kde_vals_masked + 1e-8)
                
                    color = self.get_class_color(cls)
                    label = self.get_class_label(cls)
                 
                    from matplotlib.colors import LinearSegmentedColormap
                    cmap = LinearSegmentedColormap.from_list(f'custom_{cls}', 
                                                       [(1, 1, 1, 0), color], 
                                                       N=256)
                
                    # Plot with levels based on percentiles
                    if not np.all(np.isnan(kde_vals_masked)):
                        valid_vals = kde_vals_masked[~np.isnan(kde_vals_masked)]
                        if len(valid_vals) > 0:
                            # Create 4 contour levels from 25th to 90th percentile
                            levels = np.percentile(valid_vals, [25, 50, 75, 90])
                        
                            contours = self.ax_latent.contourf(xx, yy, kde_vals_masked, 
                                                          levels=levels, alpha=alpha,
                                                          cmap=cmap, antialiased=True,
                                                          extend='max')
                        
                            # Contour lines
                            self.ax_latent.contour(xx, yy, kde_vals_masked, 
                                              levels=levels, colors=[color], 
                                              linewidths=0.5, alpha=0.8)
                
                    # Add label
                    if class_points is not None and len(class_points) > 0:
                        center_x = np.median(class_points[:, 0])
                        center_y = np.median(class_points[:, 1])
                    
                        if (x_min <= center_x <= x_max) and (y_min <= center_y <= y_max):
                            self.ax_latent.text(center_x, center_y, label,
                                           fontsize=9, fontweight='bold',
                                           ha='center', va='center',
                                           bbox=dict(boxstyle="round,pad=0.3",
                                                    facecolor='white',
                                                    alpha=0.8))
                except Exception as e:
                    print(f"Error plotting KDE for class {cls}: {e}")
    
        # Set plot limits
        self.ax_latent.set_xlim(x_min, x_max)
        self.ax_latent.set_ylim(y_min, y_max)

    
    def draw_scatter_points(self, points, labels, alpha=0.7):
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            if np.any(mask):  # Check if there are points for this label
                color = self.get_class_color(label)
                class_label = self.get_class_label(label)
                
                self.ax_latent.scatter(points[mask, 0], points[mask, 1],
                                      c=[color], alpha=alpha, s=30,
                                      edgecolors='white', linewidth=0.5,
                                      label=class_label)
    
    def get_scatter_radius(self):
        """Calculate a reasonable radius for scatter point selection."""
        # Calculate based on point density
        if self.mode == 'reconstruct_trajectories':
            points = self.train_latent_2d
        else:
            points = self.test_latent_2d
    
        if len(points) > 1:
            # Calculate average nearest neighbor distance
            distances = []
            for i in range(min(100, len(points))):
                point = points[i]
                other_points = np.delete(points, i, axis=0)
                dist = np.min(np.sqrt(np.sum((other_points - point) ** 2, axis=1)))
                distances.append(dist)
        
            avg_distance = np.mean(distances)
            return avg_distance * 0.3  # 30% of average spacing
        else:
            return 0.1  # Default radius
    
    def find_nearest_point_idx(self, points, target):
        """Find index and distance of nearest point."""
        distances = np.sqrt(np.sum((points - target) ** 2, axis=1))
        nearest_idx = np.argmin(distances)
        nearest_distance = distances[nearest_idx]
    
        return nearest_idx, nearest_distance
    
    def on_latent_click(self, event):
        if event.inaxes != self.ax_latent:
            return
        
        click_x, click_y = event.xdata, event.ydata
        self.coord_label.setText(f"Coordinates: ({click_x:.3f}, {click_y:.3f})")
        
        if self.mode == 'generate_trajectories':
            self.generate_from_point(click_x, click_y)
            self.is_test_point = False
            
        elif self.mode == 'reconstruct_trajectories':
            # Find nearest training point by direct distance calculation
            distances = np.sqrt(np.sum((self.train_latent_2d - [click_x, click_y])**2, axis=1))
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]
            
            if nearest_distance < 1.0:  # Increased threshold for easier clicking
                self.reconstruct_training_point(nearest_idx)
                click_x, click_y = self.train_latent_2d[nearest_idx]
                self.is_test_point = False
            else:
                self.status_bar.showMessage("Click closer to a training point")
                return
            
        elif self.mode == 'project_new_trajectories':
            if self.test_latent_2d is None:
                self.status_bar.showMessage("No test data available")
                return
            
            # Find nearest test point by direct distance calculation
            distances = np.sqrt(np.sum((self.test_latent_2d - [click_x, click_y])**2, axis=1))
            nearest_idx = np.argmin(distances)
            nearest_distance = distances[nearest_idx]
            
            if nearest_distance < 1.0:  # Increased threshold for easier clicking
                self.project_test_point(nearest_idx)
                click_x, click_y = self.test_latent_2d[nearest_idx]
                self.is_test_point = True
            else:
                self.status_bar.showMessage("Click closer to a test point")
                return
        '''
        # Update selected point marker
        if self.selected_point is not None:
            self.selected_point.remove()
        
        if self.is_test_point:
            marker = 's'
            color = 'black'
            size = 100
        else:
            marker = 'o'
            color = 'red'
            size = 80
        
        self.selected_point = self.ax_latent.scatter([click_x], [click_y],
                                                    c=color, marker=marker,
                                                    s=size, edgecolors='white',
                                                    linewidth=2, zorder=10)
        
        self.canvas_latent.draw()
        '''
        self.update_selected_point(click_x, click_y)


    def update_selected_point(self, x, y):
        """Safely update the selected point marker."""
        try:
            # Remove old marker if it exists and is a valid artist
            if self.selected_point is not None:
                try:
                    # Check if it's a matplotlib artist
                    if hasattr(self.selected_point, 'remove'):
                        self.selected_point.remove()
                    else:
                        # Try to remove it from the axes collections
                        if self.selected_point in self.ax_latent.collections:
                            self.selected_point.remove()
                        elif self.selected_point in self.ax_latent.lines:
                            self.selected_point.remove()
                except (AttributeError, NotImplementedError, ValueError) as e:
                    # If removal fails, just clear the axes and redraw
                    print(f"Warning: Could not remove old marker: {e}")
                    pass
        
            # Clear any existing selected points from axes
            self.clear_selected_points()
        
        except Exception as e:
            print(f"Error removing old marker: {e}")
            # Clear axes and redraw
            self.redraw_latent_space()
    
        # Create new marker
        if self.is_test_point:
            marker = 's'
            color = 'black'
            size = 100
        else:
            marker = 'o'
            color = 'red'
            size = 80
    
        self.selected_point = self.ax_latent.scatter([x], [y],
                                                c=color, marker=marker,
                                                s=size, edgecolors='white',
                                                linewidth=2, zorder=10)
    
        self.canvas_latent.draw()

    def clear_selected_points(self):
        """Clear any selected points from the axes."""
        # Remove scatter collections that look like selected points
        collections_to_remove = []
        for collection in self.ax_latent.collections:
            # Check if this looks like a selected point (small number of points, specific size)
            if (hasattr(collection, '_sizes') and 
                len(collection._sizes) == 1 and 
                collection._sizes[0] in [60, 80, 100]):
                collections_to_remove.append(collection)
    
        for collection in collections_to_remove:
            try:
                collection.remove()
            except:
                pass



    def on_slider_changed(self):
        if self.mode != 'generate_trajectories':
            return
    
        pca1_val = self.pca1_slider.value()
        pca2_val = self.pca2_slider.value()
    
        # Convert slider values to PCA coordinates
        # Get the range of PCA values from training data
        x_min, x_max = self.train_latent_2d[:, 0].min(), self.train_latent_2d[:, 0].max()
        y_min, y_max = self.train_latent_2d[:, 1].min(), self.train_latent_2d[:, 1].max()
    
        click_x = x_min + (pca1_val + 100) / 200 * (x_max - x_min)
        click_y = y_min + (pca2_val + 100) / 200 * (y_max - y_min)
    
        self.coord_label.setText(f"Coordinates: ({click_x:.3f}, {click_y:.3f})")
    
        # Use the same generate function
        self.generate_from_point(click_x, click_y)
        self.is_test_point = False
    
        # Update the marker
        if self.selected_point is not None:
            self.selected_point.remove()
    
        self.selected_point = self.ax_latent.scatter([click_x], [click_y],
                                                c='red', marker='o',
                                                s=60, edgecolors='white',
                                                linewidth=2, zorder=10)
    
        self.canvas_latent.draw()
    
    def generate_from_point(self, pca1_val, pca2_val):
        """Generate trajectory from arbitrary PCA coordinates."""
        
        try:
            print(f"\n=== Generating from point ({pca1_val:.3f}, {pca2_val:.3f}) ===")
            
            # Create a 2D point from the user click
            pca_2d_point = np.array([[pca1_val, pca2_val]])
        
            # Direct inverse transform from 2D PCA space back to latent space
            latent_vector = self.pca.inverse_transform(pca_2d_point)
            
            print(f"PCA inverse output shape: {latent_vector.shape}")
            print(f"Model decoder input should be: 50 dimensions")
            
            # Decode the latent vector
            with torch.no_grad():
                #latent_tensor = torch.FloatTensor(latent_vector)
                #generated = self.model.decode(latent_tensor).numpy()


                latent_tensor = torch.from_numpy(latent_vector).float()#.to(device)
                generated = self.model.decode(latent_tensor).cpu().numpy()[0]
            

            print(f"Decoder output shape: {generated.shape}")
            print(f"Expected output: ({self.seq_length}, {self.n_features}) = ({self.seq_length*self.n_features} flattened)")
            
            # Reshape to trajectory
            #if generated.shape[1] == self.seq_length * self.n_features:
                #trajectory = generated.reshape(self.seq_length, self.n_features)
            if len(generated) == self.seq_length * self.n_features:
                trajectory = generated.reshape(self.seq_length,self.n_features)
            else:
                # Try to reshape based on actual shape
                print(f"Warning: Output shape {generated.shape} doesn't match expected")
                  
            print(f"Final trajectory shape: {trajectory.shape}")
            
            self.current_trajectory = trajectory
        
            self.display_trajectory(trajectory, pca1_val, pca2_val, "Generated")
            self.status_bar.showMessage(f"Generated from PCA coordinates ({pca1_val:.2f}, {pca2_val:.2f})")
        
        except Exception as e:
            self.status_bar.showMessage(f"Error generating trajectory: {str(e)}")
            print(f"Error in generate_from_point: {e}")
            traceback.print_exc()
    
    def reconstruct_training_point(self, idx):
        """Reconstruct a training point using its stored latent vector."""
        original = self.train_data[idx].reshape(self.seq_length, self.n_features)
    
        # Use the pre-computed latent vector (not PCA)
        latent_vector = self.train_latent[idx].reshape(1, -1)
    
        with torch.no_grad():
            latent_tensor = torch.FloatTensor(latent_vector)
            
            # Check if model has a decoder attribute
            if hasattr(self.model, 'decoder'):
                reconstructed = self.model.decoder(latent_tensor).numpy()
            else:
                # Try to use model directly
                reconstructed = self.model(latent_tensor).numpy()
    
        # Reshape
        if reconstructed.shape[1] == self.seq_length * self.n_features:
            reconstructed = reconstructed.reshape(self.seq_length, self.n_features)
        elif reconstructed.shape[0] == 1:
            reconstructed = reconstructed[0].reshape(self.seq_length, self.n_features)
        else:
            reconstructed = reconstructed.reshape(self.seq_length, self.n_features)
    
        self.current_trajectory = reconstructed
        self.original_trajectory = original
    
        # Get the 2D coordinates for display
        pca1_val, pca2_val = self.train_latent_2d[idx]
        self.display_comparison(original, reconstructed, pca1_val, pca2_val, "Training")
    
        label = self.train_labels[idx]
        class_label = self.get_class_label(label)
        self.status_bar.showMessage(f"Training point {idx} - {class_label}")
    
    def project_test_point(self, idx):
        """Project a test point using its stored latent vector."""
        if self.test_data is None:
            self.status_bar.showMessage("No test data available")
            return
    
        original = self.test_data[idx].reshape(self.seq_length, self.n_features)
    
        # Use the pre-computed latent vector (not PCA)
        latent_vector = self.test_latent[idx].reshape(1, -1)
    
        with torch.no_grad():
            latent_tensor = torch.FloatTensor(latent_vector)
            
            # Check if model has a decoder attribute
            if hasattr(self.model, 'decoder'):
                reconstructed = self.model.decoder(latent_tensor).numpy()
            else:
                # Try to use model directly
                reconstructed = self.model(latent_tensor).numpy()
    
        # Reshape
        if reconstructed.shape[1] == self.seq_length * self.n_features:
            reconstructed = reconstructed.reshape(self.seq_length, self.n_features)
        elif reconstructed.shape[0] == 1:
            reconstructed = reconstructed[0].reshape(self.seq_length, self.n_features)
        else:
            reconstructed = reconstructed.reshape(self.seq_length, self.n_features)
    
        self.current_trajectory = reconstructed
        self.original_trajectory = original
    
        # Get the 2D coordinates for display
        pca1_val, pca2_val = self.test_latent_2d[idx]
        self.display_comparison(original, reconstructed, pca1_val, pca2_val, "Test")
    
        self.status_bar.showMessage(f"Test point {idx}")
    
    def display_trajectory(self, trajectory, pca1, pca2, source):
        self.ax_trajectory.clear()
        
        # Check if we have valid trajectory data
        if trajectory is None or len(trajectory) == 0:
            self.ax_trajectory.text(0.5, 0.5, f"No {source} trajectory data",
                                   ha='center', va='center', fontsize=12)
            self.ax_trajectory.set_title(f'{source} Trajectory\nPCA: ({pca1:.2f}, {pca2:.2f})\nNo Data')
        else:
            x_pos = trajectory[:, 0]
            y_pos = trajectory[:, 1]
            
            self.ax_trajectory.plot(x_pos, y_pos, 'b-', linewidth=2, label='Trajectory')
            self.ax_trajectory.plot(x_pos[0], y_pos[0], 'go', markersize=10, label='Start')
            self.ax_trajectory.plot(x_pos[-1], y_pos[-1], 'ro', markersize=10, label='End')
            
            if self.n_features >= 4:
                skip = max(1, self.seq_length // 20)
                for i in range(0, self.seq_length, skip):
                    self.ax_trajectory.arrow(x_pos[i], y_pos[i], 
                                           trajectory[i, 2] * 0.1, trajectory[i, 3] * 0.1,
                                           head_width=0.05, head_length=0.1,
                                           fc='orange', ec='orange', alpha=0.6)
            
            self.ax_trajectory.set_title(f'{source} Trajectory\n From X,Y Coordinates: ({pca1:.2f}, {pca2:.2f})')
        
        self.ax_trajectory.set_xlabel('X Position')
        self.ax_trajectory.set_ylabel('Y Position')
        self.ax_trajectory.grid(True, alpha=0.3)
        self.ax_trajectory.legend(loc="upper right")
        self.ax_trajectory.set_aspect('equal', adjustable='datalim')
        #self.ax_trajectory.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig_trajectory.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas_trajectory.draw()
    
    def display_comparison(self, original, reconstructed, pca1, pca2, source):
        self.ax_trajectory.clear()
        
        # Check if we have valid data
        if original is None or reconstructed is None:
            self.ax_trajectory.text(0.5, 0.5, f"No {source} data",
                                   ha='center', va='center', fontsize=12)
            self.ax_trajectory.set_title(f'{source} Reconstruction\nNo Data')
        else:
            x_orig = original[:, 0]
            y_orig = original[:, 1]
            x_recon = reconstructed[:, 0]
            y_recon = reconstructed[:, 1]
            
            self.ax_trajectory.plot(x_orig, y_orig, 'b-', linewidth=2, label='Original', alpha=0.8)
            self.ax_trajectory.plot(x_orig[0], y_orig[0], 'go', markersize=10, label='Start (Orig)')
            self.ax_trajectory.plot(x_orig[-1], y_orig[-1], 'ro', markersize=10, label='End (Orig)')
            
            self.ax_trajectory.plot(x_recon, y_recon, 'r--', linewidth=2, label='Reconstructed', alpha=0.8)
            self.ax_trajectory.plot(x_recon[0], y_recon[0], 'g^', markersize=10, label='Start (Recon)')
            self.ax_trajectory.plot(x_recon[-1], y_recon[-1], 'r^', markersize=10, label='End (Recon)')
            
            mse = np.mean((original - reconstructed) ** 2)
            
            self.ax_trajectory.set_title(f'{source} Reconstruction\n MSE: {mse:.4f}')
        
        self.ax_trajectory.set_xlabel('X Position')
        self.ax_trajectory.set_ylabel('Y Position')
        self.ax_trajectory.grid(True, alpha=0.3)
        self.ax_trajectory.legend(bbox_to_anchor=(1,1))
        self.ax_trajectory.set_aspect('equal', adjustable='datalim')
        #self.ax_trajectory.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig_trajectory.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas_trajectory.draw()
    
    def set_mode(self, mode):
        self.mode = mode
        
        if mode == 'generate_trajectories':
            self.mode_generate.setChecked(True)
            self.pca1_slider.setEnabled(True)
            self.pca2_slider.setEnabled(True)
        elif mode == 'reconstruct_trajectories':
            self.mode_reconstruct.setChecked(True)
            self.pca1_slider.setEnabled(False)
            self.pca2_slider.setEnabled(False)
        elif mode == 'project_new_trajectories':
            self.mode_project.setChecked(True)
            self.pca1_slider.setEnabled(False)
            self.pca2_slider.setEnabled(False)
        
        self.redraw_latent_space()
    
    def save_trajectory(self):
        if self.current_trajectory is not None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.original_trajectory is not None:
                np.savez(f'trajectory_{timestamp}.npz',
                        original=self.original_trajectory,
                        reconstructed=self.current_trajectory,
                        mode=self.mode,
                        pca_variance=self.explained_variance)
                self.status_bar.showMessage(f"Saved comparison to trajectory_{timestamp}.npz")
            else:
                np.save(f'trajectory_{timestamp}.npy', self.current_trajectory)
                self.status_bar.showMessage(f"Saved to trajectory_{timestamp}.npy")
    
    def reset_view(self):
        self.pca1_slider.setValue(0)
        self.pca2_slider.setValue(0)
        
        self.ax_trajectory.clear()
        self.ax_trajectory.set_xlabel('X Position')
        self.ax_trajectory.set_ylabel('Y Position')
        self.ax_trajectory.set_title('Trajectory')
        self.ax_trajectory.grid(True, alpha=0.3)
        self.canvas_trajectory.draw()
        
        if self.selected_point is not None:
            self.selected_point.remove()
            self.selected_point = None
            self.canvas_latent.draw()
        
        self.status_bar.showMessage(f"View reset - PCA Variance: {self.explained_variance:.3f}")
    
    def export_plot(self):
        self.fig_latent.savefig('latent_space.png', dpi=150, bbox_inches='tight')
        self.fig_trajectory.savefig('trajectory.png', dpi=150, bbox_inches='tight')
        self.status_bar.showMessage("Plots saved as latent_space.png and trajectory.png")