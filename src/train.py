# deep learning libraries
import torch
import torch.nn as nn

# Own libraries
from src.data import load_data
from src.datasets import PianorollGanCNNDataset
from src.gan_cnn_model import Discriminator, Generator
from src.utils import set_seed

# Other libraries
import numpy as np
import os
from tqdm import tqdm


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

torch.autograd.set_detect_anomaly(True)


"""
Ideas:
    change batch norm for layer norm
    make the discriminator worse
    explore GAN training techniques
        x one-sided label smoothing
        x feature matching
        x dropout
        -- hpp tuning (more batch size, etc)
        x initialization
        - minibatch discrimination
        - Virtual batch normalization
        - Spectral normalization
        - Try different cost functions, such as WGAN (check if use bn when changing cost function)
"""


def main():
    run = (
        max(
            int(file.split("_")[2])
            for file in os.listdir("generated")
            if "gan_cnn" in file
        )
        + 1
    )
    print(f"Run: {run}")

    batch_size: int = 128
    n_notes: int = 64
    pitch_dim: int = 88
    forward_dim: int = 256
    cond_dim: int = 256
    z_dim: int = 100

    temperature: float = 1.0

    epochs: int = 20

    lr_g: float = 0.00025
    lr_d: float = 0.0002

    dropout = 0.5

    l_1: float = 0.1
    l_2: float = 1.0

    dataloader = load_data(
        PianorollGanCNNDataset, n_notes=n_notes, batch_size=batch_size
    )

    discriminator = Discriminator(
        pitch_dim=pitch_dim, bar_length=n_notes, dropout=dropout
    ).to(device)
    generator = Generator(
        pitch_dim=pitch_dim,
        forward_dim=forward_dim,
        cond_dim=cond_dim,
        z_dim=z_dim,
        bar_length=n_notes,
        temperature=temperature,
    ).to(device)

    print(discriminator)
    print(generator)

    criterion = nn.BCEWithLogitsLoss()
    feature_criterion = nn.MSELoss()

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999)
    )
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))

    pred_noise = torch.randn(batch_size, z_dim).to(device)

    accuracy_real = 0
    accuracy_fake = 0
    n_iter = 0

    for epoch in range(epochs):
        p_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        generator.train()
        discriminator.train()
        for real, prev in p_bar:
            real = real.to(device)
            prev = prev.to(device)

            ### Train the discriminator with real data
            d_optimizer.zero_grad()

            d_real, _ = discriminator(real)

            # Add label smoothing
            d_loss_real = criterion(d_real, 0.9 * torch.ones_like(d_real))
            d_loss_real.backward(retain_graph=False)

            accuracy_real += (torch.sigmoid(d_real) > 0.5).float().mean().item()

            ### Train the discriminator with fake data
            noise = torch.randn(batch_size, z_dim).to(device)

            fake = generator(noise, prev)
            d_fake, _ = discriminator(fake.detach())

            d_loss_fake = criterion(d_fake, torch.zeros_like(d_fake))
            d_loss_fake.backward(retain_graph=False)

            accuracy_fake += (torch.sigmoid(d_fake) < 0.5).float().mean().item()

            d_loss = d_loss_real + d_loss_fake

            d_optimizer.step()

            ### Train the generator
            g_optimizer.zero_grad()

            d_fake, _ = discriminator(fake)

            g_loss = criterion(d_fake, torch.ones_like(d_fake))

            fx_loss_1 = feature_criterion(real.mean(0), fake.mean(0))
            fx_loss_1 = l_1 * fx_loss_1

            # fx_loss_2 = feature_criterion(fx_fake.mean(0), fx_real.mean(0))
            # fx_loss_2 = l_2 * fx_loss_2

            g_loss = g_loss + fx_loss_1
            g_loss.backward(retain_graph=False)

            g_optimizer.step()

            # Train the generator a second time

            g_optimizer.zero_grad()

            noise = torch.randn(batch_size, z_dim).to(device)
            fake = generator(noise, prev)

            d_fake, _ = discriminator(fake)

            g_loss = criterion(d_fake, torch.ones_like(d_fake))

            fx_loss_1 = feature_criterion(real.mean(0), fake.mean(0))
            fx_loss_1 = l_1 * fx_loss_1

            # fx_loss_2 = feature_criterion(fx_fake.mean(0), fx_real.mean(0))
            # fx_loss_2 = l_2 * fx_loss_2

            g_loss = g_loss + fx_loss_1
            g_loss.backward(retain_graph=False)

            g_optimizer.step()

            n_iter += 1

            p_bar.set_postfix(
                {
                    "D Loss": d_loss.item(),
                    "G Loss": g_loss.item(),
                    "Acc Real": accuracy_real / n_iter,
                    "Acc Fake": accuracy_fake / n_iter,
                }
            )

        accuracy_real = 0
        accuracy_fake = 0
        n_iter = 0

        # Save the model
        if epoch % 2 == 0:
            print(f"Saving model at epoch {epoch}")

            torch.save(
                {
                    "discriminator": discriminator.state_dict(),
                    "generator": generator.state_dict(),
                },
                f"checkpoints/gan_cnn_{run}_{epoch}.pt",
            )
            # generate samples and save them
            generator.eval()
            with torch.no_grad():
                fake = generator(pred_noise, prev)
                np.save(f"generated/gan_cnn_{run}_{epoch}_fake.npy", fake.cpu().numpy())
                np.save(f"generated/gan_cnn_{run}_{epoch}_real.npy", real.cpu().numpy())

    torch.save(
        {
            "discriminator": discriminator.state_dict(),
            "generator": generator.state_dict(),
        },
        f"checkpoints/gan_cnn_{run}_{epoch}.pt",
    )


if __name__ == "__main__":
    main()
