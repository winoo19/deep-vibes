import torch
import matplotlib.pyplot as plt
from typing import Dict
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
from src.utils import save_model
from src.datasets import PianorollDataset
from src.models_lstm import GeneratorLSTM, DiscriminatorLSTM, GeneratorLSTM2D, DiscriminatorLSTM2D, GANLSTM, WGANLSTM
from tqdm import tqdm
import numpy as np
import os

torch.manual_seed(42)

def define_model_gan() -> Dict[str, object]:
    """
    Function that defines the models and hyperparameters for the GAN.
    """
    # song-related hyperparameters. 
    n_steps: int = 80
    output_size: int = 88

    # model-related hyperparameters.
    lr: float = 0.0001
    epochs: int = 100
    latent_dim: int = 100 # size of the latent vector.
    hidden_size_g: int = 512 # size of the hidden state for the generator.
    hidden_size_d: int = 256 # size of the hidden state for the discriminator. 
    dropout: float = 0.3
    slope: float = 0.01
    batch_size: int = 64

    # create the generator and discriminator
    generator: torch.nn.Module = GeneratorLSTM2D(n_steps, latent_dim, hidden_size_g, output_size, dropout, slope)
    discriminator: torch.nn.Module = DiscriminatorLSTM2D(n_steps, output_size, hidden_size_d, dropout, slope)
    return generator, discriminator, latent_dim, lr, epochs, batch_size, n_steps

def get_data_loader(n_steps: int, batch_size: int):
    data_path = os.path.join("data", "npy")
    dataset = PianorollDataset(data_path, n_steps)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader


def train_gan_lstm():
    """
    Trains a GAN using a LSTM.
    """

    generator: torch.nn.Module
    discriminator: torch.nn.Module
    latent_dim: int
    lr: float
    epochs: int
    batch_size: int
    n_steps: int

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get models and hyperparameters.
    generator, discriminator, latent_dim, lr, epochs, batch_size, n_steps = define_model_gan()
    gan_lstm: torch.nn.Module = GANLSTM(generator, discriminator, device)
    # create optimizers.
    optimizer_g: torch.optim.Optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d: torch.optim.Optimizer = torch.optim.SGD(discriminator.parameters(), lr=lr)
    gan_lstm.define_optimizer(optimizer_g, optimizer_d)
    # create the dataloader.
    train_loader = get_data_loader(n_steps, batch_size)
    # train the GAN.
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for idx, real in enumerate(pbar):
            real: torch.Tensor = real.to(device)
            noise: torch.Tensor = torch.randn(batch_size, latent_dim, device=gan_lstm.device)
            fake: torch.Tensor = gan_lstm.generator(noise)
            gan_lstm.discriminator_step(real, fake)
            gan_lstm.generator_step(fake)
            pbar.set_postfix({"LG": f"{np.mean(gan_lstm.metrics['gen-loss']):.4f}", "LC": f"{np.mean(gan_lstm.metrics['discriminator-loss']):.4f}"})
        


        
def train_wgan_lstm():
    """
    Trains a WGAN using a LSTM.
    """

    generator: torch.nn.Module
    discriminator: torch.nn.Module
    latent_dim: int
    epochs: int
    batch_size: int
    n_steps: int

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get models and hyperparameters
    generator, discriminator, latent_dim, _, epochs, batch_size, n_steps = define_model_gan()

    # create the optimizers
    wgan = WGANLSTM(generator, discriminator,device)
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=0.00005)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=0.00005)
    wgan.define_optimizer(optimizer_g, optimizer_d)
    # create the dataloader
    train_loader = get_data_loader(n_steps, batch_size)
    # train the WGAN.
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for idx, real in enumerate(pbar):
            real = real.to(device)
            noise = torch.randn(batch_size, latent_dim, device = device)
            fake = generator(noise)
            wgan.discriminator_step(real, fake)
            if idx % wgan.n_critic == 0:
                wgan.generator_step(fake)
            # plot losses of the generator and discriminator
            pbar.set_postfix({"LG": f"{np.mean(wgan.metrics['gen-loss']):.4f}", "LC": f"{np.mean(wgan.metrics['discriminator-loss']):.4f}"})


if __name__ == "__main__":
    train_wgan_lstm()
    train_gan_lstm()