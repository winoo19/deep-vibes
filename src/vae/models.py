import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(torch.nn.Module):
    """
    Autoencoder model.

    Args:
        input_size (int): Size of the input tensor.
        encoder_hidden_size (int): Size of the hidden layer in the encoder.
        decoder_hidden_size (int): Size of the hidden layer in the decoder.
    """

    def __init__(
        self,
        input_size: int,
        embed_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
    ):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, encoder_hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(encoder_hidden_size, embed_size),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embed_size, decoder_hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(decoder_hidden_size, input_size),
            torch.nn.Sigmoid(),
        )

        self.de_dx: torch.Tensor = None

    def forward(self, x):
        e = self.encoder(x)
        x = self.decoder(e)
        return x, e

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class AELoss(torch.nn.Module):
    def __init__(self, loss: torch.nn.Module):
        super(AELoss, self).__init__()
        self.loss = loss

    def forward(self, x: tuple[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tuple[torch.Tensor]): Tuple with the output tensor and the encoded tensor.
            y (torch.Tensor): Target tensor.
        """
        return self.loss(x[0], y)


class LSTMVAE(torch.nn.Module):
    """
    Autoencoder model.

    Args:
        input_size (int): Size of the input tensor.
        encoder_hidden_size (int): Size of the hidden layer in the encoder.
        decoder_hidden_size (int): Size of the hidden layer in the decoder.
    """

    def __init__(
        self,
        input_size: int,
        embed_size: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        # Encoder
        self.encoder = torch.nn.GRU(
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_layers,
            batch_first=True,
        )
        self.mean_linear = torch.nn.Linear(encoder_hidden_size, embed_size)
        self.logvar_linear = torch.nn.Sequential(
            torch.nn.Linear(encoder_hidden_size, embed_size),
            torch.nn.ReLU(),
        )

        # Decoder
        self.decoder = torch.nn.GRU(
            input_size=embed_size,
            hidden_size=decoder_hidden_size,
            num_layers=decoder_layers,
            batch_first=True,
        )
        self.output_linear = torch.nn.Linear(decoder_hidden_size, input_size)
        self.sigmoid = torch.nn.Sigmoid()

        self.input_size = input_size
        self.embed_size = embed_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor. Dimensions: [batch_size, n_notes, 88]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Tuple with the output, mean, std tensors.
        """
        # Encode
        _, hn = self.encoder(x)  # [1, batch_size, encoder_hidden_size]
        hn = hn.squeeze(0)  # [batch_size, encoder_hidden_size]
        mean = self.mean_linear(hn)  # [batch_size, embed_size]
        std = self.logvar_linear(hn)  # [batch_size, embed_size]

        # Reparametrization trick
        z = mean + std * torch.randn_like(std)  # [batch_size, embed_size]

        # Decode
        z_rep = z.unsqueeze(1).repeat(
            1, x.shape[1], 1
        )  # [batch_size, n_notes, embed_size]
        outputs, _ = self.decoder(z_rep)  # [batch_size, n_notes, decoder_hidden_size]
        x_hat = self.sigmoid(self.output_linear(outputs))  # [batch_size, n_notes, 88]

        return x_hat, mean, std

    def generate(self, n_notes: int) -> torch.Tensor:
        """
        Decode the latent vector.

        Args:
            z (torch.Tensor): Latent vector.
        """
        z = torch.randn(1, self.embed_size).to(device)
        z_rep = z.unsqueeze(1).repeat(
            1, n_notes, 1
        )  # [batch_size, n_notes, embed_size]
        outputs, _ = self.decoder(z_rep)
        x_hat = self.sigmoid(self.output_linear(outputs))

        return x_hat


class CNNVAE(torch.nn.Module):
    """
    Autoencoder model.

    Args:
        n_notes (int): Number of notes in the pianoroll.
        n_features (int): Number of features in the pianoroll.
        embed_size (int): Size of the embedding.
    """

    def __init__(
        self,
        n_notes: int,
        n_features: int,
        embed_size: int,
    ):
        super().__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
        )  # [batch_size, 64, 10, 11]

        dummy_input = torch.randn(1, 1, n_notes, n_features)
        dummy_output = self.encoder(dummy_input)
        self.encoder_output_shape = dummy_output.shape
        encoder_output_size = dummy_output.view(1, -1).shape[1]

        self.mean_linear = torch.nn.Linear(encoder_output_size, embed_size)
        self.logvar_linear = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_size, embed_size),
            torch.nn.ReLU(),
        )

        # Upsample linear layer
        self.upsample_linear = torch.nn.Sequential(
            torch.nn.Linear(embed_size, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, encoder_output_size),
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                128, 128, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)
            ),
            torch.nn.LeakyReLU(),
            # Sharpen layers
            torch.nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1),
        )

        self.n_notes = n_notes
        self.n_features = n_features
        self.embed_size = embed_size

        # Use xavier initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                # torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.kaiming_normal_(m.weight, a=0.01)

        # Running mean and std buffers
        self.register_buffer("running_mean", torch.zeros(1, embed_size))
        self.register_buffer("running_std", torch.ones(1, embed_size))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor. Dimensions: [batch_size, n_notes, 88]

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Tuple with the output, mean, std tensors.
        """
        # Encode
        x = x.unsqueeze(1)  # Add channel dimension
        h1 = self.encoder(x)  # [batch_size, 64, 10, 11]
        h1 = h1.view(h1.size(0), -1)  # [batch_size, 64 * 10 * 11]
        mean = self.mean_linear(h1)  # [batch_size, embed_size]
        std = self.logvar_linear(h1)  # [batch_size, embed_size]

        # Update running mean and std
        self.running_mean = 0.99 * self.running_mean + 0.01 * mean.mean(0, keepdim=True)
        self.running_std = 0.99 * self.running_std + 0.01 * std.mean(0, keepdim=True)

        # Reparametrization trick
        z = mean + std * torch.randn_like(std)  # [batch_size, embed_size]

        # Decode
        z_up = self.upsample_linear(z).view(
            -1, *self.encoder_output_shape[1:]
        )  # [batch_size, 64, 10, 11]

        x_hat = self.decoder(z_up)  # [batch_size, 1, n_notes, n_features]
        x_hat = x_hat.squeeze(1)  # [batch_size, n_notes, n_features]

        return x_hat, mean, std

    def generate(self) -> torch.Tensor:
        """
        Decode the latent vector.

        Args:
            z (torch.Tensor): Latent vector.
        """
        # Use running mean and std
        z = (
            self.running_mean
            + torch.randn(1, self.embed_size).to(device) * self.running_std
        )
        # z = torch.randn(1, self.embed_size, dtype=torch.double).to(device)
        z_up = self.upsample_linear(z).view(
            -1, *self.encoder_output_shape[1:]
        )  # [batch_size, 64, 10, 11]
        x_hat = self.decoder(z_up)  # [batch_size, 1, n_notes, n_features]
        x_hat = x_hat.squeeze(1)  # [batch_size, n_notes, n_features]

        return torch.sigmoid(x_hat)


class VAELoss(torch.nn.Module):
    """
    VAE loss function.
    """

    def __init__(self, gamma: float = 1e-3):
        super(VAELoss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.gamma = gamma

    def forward(self, x: tuple[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tuple[torch.Tensor]): Tuple with the output, mean and std tensors.
            y (torch.Tensor): Target tensor.
        """
        reconstruction_loss = self.bce_loss(x[0], y)
        mean, std = x[1], x[2]
        kl_divergence = 0.5 * torch.mean(
            mean**2 + std**2 - 2 * torch.log(std + 1e-8) - 1
        )

        return (
            reconstruction_loss + self.gamma * kl_divergence,
            reconstruction_loss,
            kl_divergence,
        )


class GammaScheduler:
    """
    Scheduler for the gamma parameter in the VAE loss.
    """

    def __init__(
        self,
        loss: VAELoss,
        zero_epochs: int,
        min_exponent: int,
        max_exponent: int,
        real_epochs: int,
    ):
        self.loss = loss
        self.zero_epochs = zero_epochs
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent
        self.real_epochs = real_epochs

        # assert self.min_exponent < self.max_exponent

        self.increase_epochs = max_exponent - min_exponent + 1
        self._step = 0

        self.step()

    def reset(self):
        """
        Reset the scheduler.
        """
        self._step = 0
        self.step()

    def step(self) -> None:
        """
        Update the gamma parameter of the loss.
        """
        if self._step < self.zero_epochs:
            self.loss.gamma = 0.0
        elif self._step < self.zero_epochs + self.increase_epochs:
            self.loss.gamma = 10 ** (
                self.min_exponent + (self._step - self.zero_epochs)
            )
        else:
            self.loss.gamma = 10**self.max_exponent

        self._step = (self._step + 1) % (
            self.zero_epochs + self.increase_epochs + self.real_epochs
        )

        return None

    def plot(self, n_cycles: int) -> None:
        """
        Plot the gamma values.
        """

        gamma_values = []
        for _ in range(n_cycles):
            for _ in range(self.zero_epochs + self.increase_epochs + self.real_epochs):
                gamma_values.append(self.loss.gamma)
                self.step()

        gamma_values.append(self.loss.gamma)

        plt.plot(gamma_values)
        plt.scatter(range(len(gamma_values)), gamma_values)
        plt.xlabel("Step")
        plt.ylabel("Gamma")
        plt.title("Exponential Cyclic Annealing")
        plt.show()

        self.reset()

        return None


def main():
    gamma_scheduler = GammaScheduler(
        loss=VAELoss(),
        zero_epochs=1,
        min_exponent=-4,
        max_exponent=-2,
        real_epochs=1,
    )

    gamma_scheduler.plot(n_cycles=3)


if __name__ == "__main__":
    main()
