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


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main():
    batch_size: int = 64
    n_notes: int = 16
    pitch_dim: int = 128
    cond_dim: int = 16
    z_dim: int = 100

    epochs: int = 20

    lr: float = 0.0002

    l_1: float = 0.1
    l_2: float = 0.01

    dataloader = load_data(
        PianorollGanCNNDataset, n_notes=n_notes, batch_size=batch_size
    )

    discriminator = Discriminator(pitch_dim=pitch_dim).to(device)
    generator = Generator(pitch_dim=pitch_dim, cond_dim=cond_dim, z_dim=z_dim).to(
        device
    )

    print(discriminator)
    print(generator)

    criterion = nn.BCEWithLogitsLoss()
    feature_criterion = nn.MSELoss()

    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
    )
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    pred_noise = torch.randn(batch_size, z_dim).to(device)

    for epoch in range(epochs):
        for i, (real, prev) in enumerate(dataloader):
            real = real.to(device)
            prev = prev.to(device)
            fake_noise = torch.randn(batch_size, z_dim).to(device)

            # Train the discriminator with real data
            d_optimizer.zero_grad()

            d_real, fx_real = discriminator(real)

            d_loss_real = criterion(d_real, torch.ones_like(d_real))
            d_loss_real.backward(retain_graph=True)

            # Train the discriminator with fake data
            fake = generator(fake_noise, prev)
            d_fake, _ = discriminator(fake.detach())

            d_loss_fake = criterion(d_fake, torch.zeros_like(d_fake))
            d_loss_fake.backward(retain_graph=True)

            d_loss = d_loss_real + d_loss_fake

            d_optimizer.step()

            # Train the generator
            g_optimizer.zero_grad()

            d_fake, fx_fake = discriminator(fake)

            g_loss = criterion(d_fake, torch.ones_like(d_fake))

            fx_loss_1 = feature_criterion(fx_fake, fx_real)
            fx_loss_1 = l_1 * fx_loss_1

            fx_loss_2 = feature_criterion(real, fake)
            fx_loss_2 = l_2 * fx_loss_2

            g_loss = g_loss + fx_loss_1 + fx_loss_2
            g_loss.backward()

            g_optimizer.step()

            # Train the generator a second time
            g_optimizer.zero_grad()

            d_fake, fx_fake = discriminator(fake)

            g_loss = criterion(d_fake, torch.ones_like(d_fake))

            fx_loss_1 = feature_criterion(fx_fake, fx_real)
            fx_loss_1 = l_1 * fx_loss_1

            fx_loss_2 = feature_criterion(real, fake)
            fx_loss_2 = l_2 * fx_loss_2

            g_loss = g_loss + fx_loss_1 + fx_loss_2
            g_loss.backward()

            g_optimizer.step()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Batch [{i}/{len(dataloader)}], "
                    f"D Loss: {d_loss.item()}, G Loss: {g_loss.item()}"
                )

        # Save the model
        if epoch % 2 == 0:
            torch.save(
                {
                    "discriminator": discriminator.state_dict(),
                    "generator": generator.state_dict(),
                },
                f"checkpoints/gan_cnn_{epoch}.pt",
            )
            # generate samples and save them
            generator.eval()
            with torch.no_grad():
                fake = generator(pred_noise, prev)
                np.save(f"generated/gan_cnn_{epoch}.npy", fake.cpu().numpy())
                np.save(f"generated/gan_cnn_{epoch}_real.npy", real.cpu().numpy())


if __name__ == "__main__":
    main()
