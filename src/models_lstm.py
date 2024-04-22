import torch
from typing import Dict


class NetLayer(torch.nn.Module):
    """
    This class represents a block of layers that can be used in the generator 
    and discriminator models.
    """

    def __init__(self, input_size: int , hidden_size: int , dropout: float, slope: float) -> None:
        """
        Initializes the LayerBlock.

        Args:
            input_size (int): The size of the input tensor.
            hidden_size (int): The size of the hidden state.
            dropout (float): The dropout rate.
            slope (float): The slope of the LeakyReLU activation function.
        """

        super().__init__()
        self.lstm: torch.nn.LSTM = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.drop: torch.nn.Dropout = torch.nn.Dropout(dropout)
        self.leaky: torch.nn.LeakyReLU = torch.nn.LeakyReLU(slope)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x, _ = self.lstm(x)
        x = self.drop(x)
        x = self.leaky(x)

        return x


class UpSample(torch.nn.Module):
    """
    This class represents a layer that upsamples the input tensor 
    to a sequence of a certain length and dimension.
    """
    def __init__(self, seq_length: int, latent_dim: int) -> None:
        """
        Initializes the UpSample.
        """
        super().__init__()
        self.seq_length: int = seq_length
        self.latent_dim: int = latent_dim
        self.conv_transpose: torch.nn.Module = torch.nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=(seq_length, 1)
                                                                                 , stride=(seq_length, 1), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.latent_dim, 1, 1)  
        x = self.conv_transpose(x)
        x = x.view(-1, self.seq_length, self.latent_dim)  
        return x



class TransposeLayer(torch.nn.Module):
    """
    This class represents a layer that transposes the input tensor.
    """
    def __init__(self, dims: tuple) -> None:
        """
        Initializes the TransposeLayer.
        """
        super().__init__()
        self.dims: tuple = dims
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return x.transpose(*self.dims)



class GeneratorLSTM(torch.nn.Module):
    """
    This class represents a generator model using an LSTM.

    Attributes:
        lstm (torch.nn.LSTM): The LSTM layer.
        linear (torch.nn.Linear): The linear layer.
    """
    dropout: float
    slope: float
    def __init__(self, seq_length: int, input_size: int, hidden_size: int, output_size: int, dropout=0.3, slope=0.01) -> None:
        """
        Initializes the GeneratorLSTM

        Args:
            input_size (int): The size of the latent vector.
            hidden_size (int): The size of the hidden state.
            output_size (int): The size of the output.
            dropout (float): The dropout rate.
            slope (float): The slope of the LeakyReLU activation function.
        """

        super().__init__()

        self.net: torch.nn.Sequential = torch.nn.Sequential(
            UpSample(seq_length, input_size),
            NetLayer(input_size, hidden_size, dropout, slope),
            NetLayer(hidden_size, hidden_size, dropout, slope),
            NetLayer(hidden_size, hidden_size, dropout, slope),
            torch.nn.Linear(hidden_size, output_size),
            torch.nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the generator.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        x = self.net(x)
        return x
    


class DiscriminatorLSTM(torch.nn.Module): 
    """
    This class represents a discriminator model using an LSTM.

    Attributes:
        lstm (torch.nn.LSTM): The LSTM layer.
        linear (torch.nn.Linear): The linear layer.
    """
    dropout: float
    slope: float
    def __init__(self, input_size: int, hidden_size: int, dropout=0.3, slope=0.01) -> None:
        """
        Initializes the DiscriminatorLSTM

        Args:
            input_size (int): The size of the input tensor.
            hidden_size (int): The size of the hidden state.
            output_size (int): The size of the output.
        """

        super().__init__()

        self.net = torch.nn.Sequential(
            NetLayer(input_size, hidden_size, dropout, slope),
            NetLayer(hidden_size, hidden_size, dropout, slope),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LeakyReLU(slope),
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the discriminator.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        return self.net(x).mean(dim = 1)

    


class GeneratorLSTM2D(torch.nn.Module):
    """
    This class represents a generator model using an LSTM2D.

    Attributes:
        lstm (torch.nn.LSTM): The LSTM layer.
        linear (torch.nn.Linear): The linear layer.
    """
    dropout: float
    slope: float
    def __init__(self, seq_length: int, input_size: int, hidden_size: int, output_size: int, dropout=0.3, slope=0.01) -> None:
        """
        Initializes the GeneratorLSTM

        Args:
            input_size (int): The size of the latent vector.
            hidden_size (int): The size of the hidden state.
            output_size (int): The size of the output.
            dropout (float): The dropout rate.
            slope (float): The slope of the LeakyReLU activation function.
        """

        super().__init__()

        self.up: torch.nn.Module = UpSample(seq_length, input_size)
        self.netV: torch.nn.Sequential = torch.nn.Sequential(
            NetLayer(input_size, hidden_size, dropout, slope),
            NetLayer(hidden_size, hidden_size, dropout, slope),
            NetLayer(hidden_size, hidden_size, dropout, slope),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, output_size)
        )
        self.netH: torch.nn.Sequential = torch.nn.Sequential(
            TransposeLayer((2, 1)),
            NetLayer(seq_length, hidden_size, dropout, slope),
            NetLayer(hidden_size, hidden_size, dropout, slope),
            NetLayer(hidden_size, hidden_size, dropout, slope),
            torch.nn.Linear(hidden_size, seq_length),
            TransposeLayer((2, 1)),
            torch.nn.Linear(input_size, output_size)
        )
        self.linear_cat: torch.nn.Module = torch.nn.Linear(2 * output_size, output_size)
        self.sigmoid: torch.nn.Module = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the generator.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        x = self.up(x)
        x_v: torch.Tensor = self.netV(x)
        x_h: torch.Tensor = self.netH(x)
        x = torch.cat((x_h, x_v), dim=2)
        x = self.linear_cat(x)

        return self.sigmoid(x)


class DiscriminatorLSTM2D(torch.nn.Module): 
    """
    This class represents a discriminator model using an LSTM.

    Attributes:
        lstm (torch.nn.LSTM): The LSTM layer.
        linear (torch.nn.Linear): The linear layer.
    """
    dropout: float
    slope: float
    def __init__(self, seq_length: int, input_size: int, hidden_size: int, dropout=0.3, slope=0.01) -> None:
        """
        Initializes the DiscriminatorLSTM

        Args:
            input_size (int): The size of the input tensor.
            hidden_size (int): The size of the hidden state.
            output_size (int): The size of the output.
        """

        super().__init__()
        self.netV: torch.nn.Sequential = torch.nn.Sequential(
            NetLayer(input_size, hidden_size, dropout, slope),
            NetLayer(hidden_size, hidden_size // 2, dropout, slope),
            NetLayer(hidden_size // 2, hidden_size // 4, dropout, slope),
            torch.nn.Linear(hidden_size // 4, hidden_size // 4)
        )
        self.netH: torch.nn.Sequential = torch.nn.Sequential(
            TransposeLayer((2, 1)),
            NetLayer(seq_length, hidden_size, dropout, slope),
            NetLayer(hidden_size, hidden_size // 2, dropout, slope),
            NetLayer(hidden_size // 2, hidden_size // 4, dropout, slope),
            torch.nn.Linear(hidden_size // 4, seq_length),
            TransposeLayer((2, 1)),
            torch.nn.Linear(input_size, hidden_size // 4)
        )
        self.mainnet: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.LeakyReLU(slope),
            torch.nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the discriminator.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """ 
        x_v: torch.Tensor = self.netV(x)
        x_h: torch.Tensor = self.netH(x)
        x = torch.cat((x_h, x_v), dim=2)
        x = self.mainnet(x)
        return x
    
    
class GANLSTM(torch.nn.Module):
    """
    This class represents a GAN model (https://arxiv.org/abs/1406.2661)
    using LSTMs.

    Attributes:
        generator (GeneratorLSTM): The generator model.
        discriminator (DiscriminatorLSTM): The discriminator model.
        criterion (torch.nn.BCELoss): The binary cross entropy loss function.
        device (torch.device): The device to run the model on.
    """
    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module, device) -> None:
        """
        Initializes the GANLSTM.

        Args:
            generator (torch.nn.Module): The generator model.
            discriminator (torch.nn.Module): The discriminator model.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.generator: torch.nn.Module = generator
        self.discriminator: torch.nn.Module = discriminator
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.device = device 
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.metrics: Dict[list[float], list[float]] = {"gen-loss": [], "discriminator-loss": []}   
    def define_optimizer(self, optimizer_g: torch.optim.Optimizer, optimizer_d: torch.optim.Optimizer) -> None:
        """
        Defines the optimizers for the generator and discriminator models.
        """ 

        self._optimizer_g: torch.optim.Optimizer = optimizer_g
        self._optimizer_d: torch.optim.Optimizer = optimizer_d

    def discriminator_step(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        """
        Performs a step for the discriminator model.
        """
        self._optimizer_d.zero_grad()

        real_pred: torch.Tensor = self.discriminator(real)
        fake_pred: torch.Tensor = self.discriminator(fake.detach())

        loss_real: torch.Tensor = self.criterion(real_pred, torch.ones_like(real_pred))
        loss_fake: torch.Tensor = self.criterion(fake_pred, torch.zeros_like(fake_pred))

        loss_d = loss_real + loss_fake
        loss_d.backward()
        self._optimizer_d.step()

        self.metrics["discriminator-loss"]+= [loss_d.item()]
    def generator_step(self, fake: torch.Tensor) -> None:
        self._optimizer_g.zero_grad()

        fake_pred: torch.Tensor = self.discriminator(fake)

        loss_g = self.criterion(fake_pred, torch.ones_like(fake_pred))
        loss_g.backward()
        self._optimizer_g.step()

        self.metrics["gen-loss"] += [loss_g.item()]
        

class WGANLSTM(GANLSTM):
    """
    Wasserstein Generative Adversarial Network.
    Uses Wasserstein distance for training GAN and
    gradient clipping to enforce 1-Lipschitz continuity.
    http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf
    """
    n_critic: int
    c: float
    def __init__(self, generator: torch.nn.Module, discriminator: torch.nn.Module, device, n_critic = 5, c = 0.01) -> None:
        """
        Initializes the WGANLSTM.
        Args:
            generator (torch.nn.Module): The generator model.
            discriminator (torch.nn.Module): The discriminator model.
            device (torch.device): The device to run the model on.
            n_critic (int): The number of critic steps.
            c (float): The clipping parameter.
        """
        super().__init__(generator, discriminator, device)
        self.n_critic: int = n_critic
        self.c: float = c

    def discriminator_step(self, real: torch.Tensor, fake: torch.Tensor) -> None:
        self._optimizer_d.zero_grad()

        real_logits: torch.Tensor = self.discriminator(real)
        fake_logits: torch.Tensor = self.discriminator(fake.detach())

        loss = -(real_logits.mean() - fake_logits.mean())
        loss.backward(retain_graph=True)
        self._optimizer_d.step()

        #  Weight clipping
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.c, self.c)

        self.metrics["discriminator-loss"] += [loss.item()]
    def generator_step(self, fake: torch.Tensor) -> None:
        self._optimizer_g.zero_grad()

        fake_logits = self.discriminator(fake)
        loss = -fake_logits.mean().view(-1)
        loss.backward()
        self._optimizer_g.step()

        self.metrics["gen-loss"] += [loss.item()]





