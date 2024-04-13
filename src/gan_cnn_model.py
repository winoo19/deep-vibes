# deep learning libraries
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    This class represents the discriminator of the GAN.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        fc (nn.Linear): The fully connected layer.
    """

    def __init__(self, pitch_dim: int = 128) -> None:
        """
        Initializes the Discriminator.
        """

        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 14, kernel_size=(2, pitch_dim), stride=2)
        self.conv2 = nn.Conv2d(14, 77, kernel_size=(4, 1), stride=2)
        self.bn1 = nn.BatchNorm2d(77)

        self.l1 = nn.Linear(231, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.l2 = nn.Linear(1024, 1)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): The input tensor. Shape (batch_size, bar_length, n_pitches).

        Returns:
            torch.Tensor: The output tensor. Shape (batch_size).
            torch.Tensor: The feature tensor. Shape (batch_size, 14, bar_length/2, 1).
        """

        batch_size = x.shape[0]

        x = x.unsqueeze(1)  # (batch_size, 1, bar_length, n_pitches)

        x = self.lrelu(self.conv1(x))  # (batch_size, 14, bar_length/2, 1)

        fx = x.clone()

        x = self.lrelu(self.bn1(self.conv2(x)))  # (batch_size, 77, (bar_length-4)/4, 1)

        x = x.view(batch_size, -1)  # (batch_size, 231)

        x = self.lrelu(self.bn2(self.l1(x)))  # (batch_size, 1024)

        x = self.l2(x)  # (batch_size, 1)

        return x, fx


class Generator(nn.Module):
    """
    This class represents the generator of the GAN.

    Attributes:
    """

    def __init__(
        self, pitch_dim: int = 128, forward_dim: int = 256, cond_dim: int = 256, z_dim: int = 100
    ) -> None:
        """
        Initializes the Generator.
        """

        super(Generator, self).__init__()

        self.pitch_dim: int = pitch_dim
        self.forward_dim: int = forward_dim
        self.cond_dim: int = cond_dim
        self.concat_dim: int = self.forward_dim + self.cond_dim
        self.z_dim: int = z_dim

        self.fc1 = nn.Linear(self.z_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, self.forward_dim * 2)
        self.bn2 = nn.BatchNorm1d(self.forward_dim * 2)

        self.bn3 = nn.BatchNorm2d(self.forward_dim)
        self.bn4 = nn.BatchNorm2d(self.forward_dim)
        self.bn5 = nn.BatchNorm2d(self.forward_dim)

        self.deconv1 = nn.ConvTranspose2d(
            self.concat_dim, self.forward_dim, kernel_size=(2, 1), stride=2
        )
        self.deconv2 = nn.ConvTranspose2d(
            self.concat_dim, self.forward_dim, kernel_size=(2, 1), stride=2
        )
        self.deconv3 = nn.ConvTranspose2d(
            self.concat_dim, self.forward_dim, kernel_size=(2, 1), stride=2
        )
        self.deconv4 = nn.ConvTranspose2d(
            self.concat_dim, 1, kernel_size=(1, self.pitch_dim), stride=(1, 2)
        )

        self.conv1 = nn.Conv2d(
            1, self.cond_dim, kernel_size=(1, self.pitch_dim), stride=(1, 2)
        )
        self.conv2 = nn.Conv2d(
            self.cond_dim, self.cond_dim, kernel_size=(2, 1), stride=2
        )
        self.conv3 = nn.Conv2d(
            self.cond_dim, self.cond_dim, kernel_size=(2, 1), stride=2
        )
        self.conv4 = nn.Conv2d(
            self.cond_dim, self.cond_dim, kernel_size=(2, 1), stride=2
        )

        self.bn_prev1 = nn.BatchNorm2d(self.cond_dim)
        self.bn_prev2 = nn.BatchNorm2d(self.cond_dim)
        self.bn_prev3 = nn.BatchNorm2d(self.cond_dim)
        self.bn_prev4 = nn.BatchNorm2d(self.cond_dim)
        self.lrelu = nn.LeakyReLU(0.2)

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
            batch_size, self.forward_dim, 2, 1
        )  # (batch_size, forward_dim, 2, 1)
        z = torch.cat((z, prev_4), 1)  # (batch_size, concat_dim, 2, 1)

        x = torch.relu(self.bn3(self.deconv1(z)))  # (batch_size, forward_dim, 4, 1)
        x = torch.cat((x, prev_3), 1)  # (batch_size, concat_dim, 4, 1)

        x = torch.relu(self.bn4(self.deconv2(x)))  # (batch_size, forward_dim, 8, 1)
        x = torch.cat((x, prev_2), 1)  # (batch_size, concat_dim, 8, 1)

        x = torch.relu(self.bn5(self.deconv3(x)))  # (batch_size, forward_dim, 16, 1)
        x = torch.cat((x, prev_1), 1)  # (batch_size, concat_dim, 16, 1)

        x = torch.sigmoid(self.deconv4(x))  # (batch_size, 1, bar_length, pitch_dim)

        return x.squeeze(1)
