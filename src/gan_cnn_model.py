# deep learning libraries
import torch
import torch.nn as nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Smoothing(nn.Module):
    """
    This class represents the smoothing layer.

    Attributes:
        alpha (float): The alpha value.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initializes the Smoothing layer.
        """
        super(Smoothing, self).__init__()

        self.kernel: torch.Tensor = (
            torch.tensor([alpha, 1.0 - alpha, 0.0]).float().view(1, 1, 3).to(device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the smoothing layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The smoothed tensor.
        """
        return nn.functional.conv2d(x, self.kernel.unsqueeze(0), padding=(0, 1))


class Discriminator(nn.Module):
    """
    This class represents the discriminator of the GAN.
    """

    def __init__(
        self, pitch_dim: int = 128, bar_length: int = 16, dropout: float = 0.5
    ):
        """
        Initializes the Discriminator.

        Args:
            pitch_dim (int): The number of pitches.
            bar_length (int): The length of the bar.
            dropout (float): The dropout rate.
        """

        super(Discriminator, self).__init__()

        self.pitch_dim: int = pitch_dim
        self.bar_length: int = bar_length

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, self.pitch_dim), stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 1), stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.l1 = nn.Linear((self.bar_length - 4) // 4 * 64, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.l2 = nn.Linear(1024, 1)

        self.lrelu = nn.LeakyReLU(0.2)

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): The input tensor. Shape (batch_size, bar_length, n_pitches).

        Returns:
            torch.Tensor: The output tensor. Shape (batch_size).
        """

        batch_size = x.shape[0]

        x = x.unsqueeze(1)  # (batch_size, 1, bar_length, n_pitches)

        x = self.lrelu(self.conv1(x))  # (batch_size, 14, bar_length/2, 1)

        x = self.dropout(x)

        x = self.lrelu(self.bn1(self.conv2(x)))  # (batch_size, 77, (bar_length-4)/4, 1)

        x = self.dropout(x)

        x = x.view(batch_size, -1)  # (batch_size, 231)

        x = self.lrelu(self.bn2(self.l1(x)))  # (batch_size, 1024)

        x = self.l2(x)  # (batch_size, 1)

        return x

    def get_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the feature tensor of the discriminator.

        Args:
            x (torch.Tensor): The input tensor. Shape (batch_size, bar_length, n_pitches).

        Returns:
            torch.Tensor: The feature tensor. Shape (batch_size, 14, bar_length/2, 1).
        """

        x = x.unsqueeze(1)

        x = self.lrelu(self.conv1(x))

        return x.view(x.shape[0], -1)

    def reset_parameters(self):
        """
        Reset the parameters of the discriminator.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    """
    This class represents the generator of the GAN.
    """

    def __init__(
        self,
        pitch_dim: int = 88,
        forward_dim: int = 256,
        cond_dim: int = 256,
        z_dim: int = 100,
        bar_length: int = 16,
        temperature: float = 1.0,
        alpha: float = 0.1,
    ) -> None:
        """
        Initializes the Generator.

        Args:
            pitch_dim (int): The number of pitches.
            forward_dim (int): The forward dimension.
            cond_dim (int): The condition dimension.
            z_dim (int): The noise dimension.
            bar_length (int): The length of the bar.
            temperature (float): The temperature of the sigmoid function.
            alpha (float): The smoothing alpha value.
        """

        super(Generator, self).__init__()

        self.pitch_dim: int = pitch_dim
        self.forward_dim: int = forward_dim
        self.cond_dim: int = cond_dim
        self.concat_dim: int = self.forward_dim + self.cond_dim
        self.z_dim: int = z_dim
        self.bar_length: int = bar_length
        self.temperature: float = temperature

        self.fc1 = nn.Linear(self.z_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, self.forward_dim * self.bar_length // 8)
        self.bn2 = nn.BatchNorm1d(self.forward_dim * self.bar_length // 8)

        self.bn3 = nn.BatchNorm2d(self.forward_dim)
        self.bn4 = nn.BatchNorm2d(self.forward_dim * 2)
        self.bn5 = nn.BatchNorm2d(self.forward_dim * 2)

        self.deconv1 = nn.ConvTranspose2d(
            self.concat_dim, self.forward_dim, kernel_size=(2, 1), stride=2
        )
        self.deconv2 = nn.ConvTranspose2d(
            self.concat_dim, self.forward_dim * 2, kernel_size=(2, 1), stride=2
        )
        self.deconv3 = nn.ConvTranspose2d(
            self.concat_dim * 2, self.forward_dim * 2, kernel_size=(2, 1), stride=2
        )
        self.deconv4 = nn.ConvTranspose2d(
            self.concat_dim * 2, 1, kernel_size=(1, self.pitch_dim), stride=(1, 2)
        )

        self.conv1 = nn.Conv2d(
            1, self.cond_dim * 2, kernel_size=(1, self.pitch_dim), stride=(1, 2)
        )
        self.conv2 = nn.Conv2d(
            self.cond_dim * 2, self.cond_dim * 2, kernel_size=(2, 1), stride=2
        )
        self.conv3 = nn.Conv2d(
            self.cond_dim * 2, self.cond_dim, kernel_size=(2, 1), stride=2
        )
        self.conv4 = nn.Conv2d(
            self.cond_dim, self.cond_dim, kernel_size=(2, 1), stride=2
        )

        self.bn_prev1 = nn.BatchNorm2d(self.cond_dim * 2)
        self.bn_prev2 = nn.BatchNorm2d(self.cond_dim * 2)
        self.bn_prev3 = nn.BatchNorm2d(self.cond_dim)
        self.bn_prev4 = nn.BatchNorm2d(self.cond_dim)

        self.lrelu = nn.LeakyReLU(0.2)
        self.smoothing = Smoothing(alpha)

        self.reset_parameters()

    def forward(self, z: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
            x_prev (torch.Tensor): The previous bar. Shape (batch_size, bar_length, pitch_dim).
            z (torch.Tensor): The noise. Shape (batch_size, z_dim).

        Returns:
            torch.Tensor: The output tensor. Shape (batch_size, bar_length, pitch_dim).
        """
        batch_size = x_prev.shape[0]
        x_prev = x_prev.unsqueeze(1)  # (batch_size, 1, bar_length, pitch_dim)

        prev_1 = self.lrelu(
            self.bn_prev1(self.conv1(x_prev))
        )  # (batch_size, cond_dim, bar_length, 1)
        prev_2 = self.lrelu(
            self.bn_prev2(self.conv2(prev_1))
        )  # (batch_size, cond_dim, bar_length/2, 1)
        prev_3 = self.lrelu(
            self.bn_prev3(self.conv3(prev_2))
        )  # (batch_size, cond_dim, bar_length/4, 1)
        prev_4 = self.lrelu(
            self.bn_prev4(self.conv4(prev_3))
        )  # (batch_size, cond_dim, bar_length/8, 1)

        z = torch.relu(self.bn1(self.fc1(z)))  # (batch_size, 1024)
        z = torch.relu(self.bn2(self.fc2(z)))  # (batch_size, forward_dim * 2)
        z = z.view(
            batch_size, self.forward_dim, self.bar_length // 8, 1
        )  # (batch_size, forward_dim, bar_lenght, 1)
        z = torch.cat((z, prev_4), 1)  # (batch_size, concat_dim, 2, 1)

        x = torch.relu(self.bn3(self.deconv1(z)))  # (batch_size, forward_dim, 4, 1)
        x = torch.cat((x, prev_3), 1)  # (batch_size, concat_dim, 4, 1)

        x = torch.relu(self.bn4(self.deconv2(x)))  # (batch_size, forward_dim, 8, 1)
        x = torch.cat((x, prev_2), 1)  # (batch_size, concat_dim, 8, 1)

        x = torch.relu(self.bn5(self.deconv3(x)))  # (batch_size, forward_dim, 16, 1)
        x = torch.cat((x, prev_1), 1)  # (batch_size, concat_dim, 16, 1)

        x = torch.sigmoid(
            self.deconv4(x) / self.temperature
        )  # (batch_size, 1, bar_length, pitch_dim)

        return x.squeeze(1)

    def reset_parameters(self):
        """
        Reset the parameters of the generator.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
