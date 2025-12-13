
# Trajectory VAE Toolbox

A modular toolbox for visualizing and exploring an autoencoder's latent space

The pipeline for this working example was developed for training/testing different trajectory shapes:

![Training pipeline](VAEpipeline.png)


Note that the hyperparams for this VAE has not been optimized, so the reconstruction error could be improved with tuning. Also note that this pipeline is projecting a compressed latent space to 2D using PCA, so the ability to reconstruct/ generate new trajectories will be constrained by the quality of your PCA reduction. In this example, the PCA only accounts for 70% of the latent space variance, so the generated trajectories are not fantastic. 

https://github.com/user-attachments/assets/9ca2f607-5a4b-4abc-a2d8-02acee4c48e2

<video src="
https://github.com/user-attachments/assets/9ca2f607-5a4b-4abc-a2d8-02acee4c48e2" controls="controls" style="max-width: 730px;"></video>

## Features

- **VAE Example with Training**: Learn compact latent representations of trajectories or load in previously trained model
- **Interactive Visualization**: Run the GUI to get a 2D latent representatino. Click on points in 2D latent space to generate trajectories or reconstruct existing
- To Do: **Hyperparameter Optimization**: Automated optimization with Optuna
- To Do: **Multi-dataset Support**: Works with various trajectory data formats
- **Diagnostic Tools**: Comprehensive plotting and analysis functions

## Installation

```bash
git clone https://github.com/JaleesaHoule/LatentSpaceExplorer.git
cd LatentSpaceExplorer
pip install -r requirements.txt
```

## Run the example

```bash
example.ipynb
```
