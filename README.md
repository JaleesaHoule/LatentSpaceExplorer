
# Latent Space Explorer

A modular toolbox for visualizing and exploring an autoencoder's latent space

The pipeline for this working example was developed for training/testing different trajectory shapes:

![Training pipeline](VAEpipeline.png)



 **This tool was built for assessing various latent space attributes of an ongoing project which is currently under review. Full information on the methods for our pipeline will be shared upon publication.**

Note that the hyperparams for this VAE have not been optimized and could be much improved with tuning. Also note that this pipeline is projecting an already compressed latent space into 2D using PCA, so the ability to reconstruct/ generate new trajectories will be constrained by the quality of your PCA reduction. In this example, going to a 2D representation allows for 70% explained variance of the latent space, so newly generated trajectories are not fantastic. A much better approach would be to estimate each class distribution and then generate samples from those distributions, but that lacks the ability to control exactly from where in the distribution space you sample. 



# Quick Demo:



<video src="https://github.com/user-attachments/assets/02fc6666-78cc-4ac0-895a-3e683173845e" controls="controls" style="max-width: 730px;"></video>

## Features

- **VAE Example with Training**: Learn compact latent representations of trajectories or load in previously trained model
- **Interactive Visualization**: Run the GUI to get a 2D latent representatino. Click on points in 2D latent space to generate trajectories or reconstruct existing
- **Diagnostic Tools**: Typical plotting and analysis functions
- To Do: **Hyperparameter Optimization**: Automated optimization with Optuna
- To Do: **Multi-dataset Support**: Add example with CNN architecture + images


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
