import pypianoroll as ppr
import numpy as np
from src.midi import matrix2pianoroll

import matplotlib.pyplot as plt

# deep learning libraries
import torch

# Own libraries
from src.utils import set_seed


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def show(file: str = "generated\gan_cnn_9_12"):
    real: np.ndarray = np.load(f"{file}_real.npy")
    fake: np.ndarray = np.load(f"{file}_fake.npy")
    prev: np.ndarray = np.load(f"{file}_prev.npy")
    # dataloader = load_data(PianorollGanCNNDataset, n_notes=64, batch_size=128)

    real_pr = [matrix2pianoroll(track.T) for track in real]
    fake_pr = [matrix2pianoroll(track.T) for track in fake]
    prev_pr = [matrix2pianoroll(track.T) for track in prev]

    # Plot the real and fake matrices side by side
    n = 5

    for i in range(n):
        ax = plt.figure(figsize=(8, 3)).add_subplot(111)
        ppr.plot_pianoroll(ax, prev_pr[i], xtick="auto")
        plt.title("Prev")
        plt.show()

        ax = plt.figure(figsize=(8, 3)).add_subplot(111)
        ppr.plot_pianoroll(ax, fake_pr[i], xtick="auto")
        plt.title("Fake")
        plt.show()

        ax = plt.figure(figsize=(8, 3)).add_subplot(111)
        ppr.plot_pianoroll(ax, real_pr[i], xtick="auto")
        plt.title("Real")
        plt.show()


def load_model(path: str):
    model = torch.load(path)

    gen = model["generator"]

    gen.load_state_dict(model["generator_params"])

    for param in gen.parameters():
        param.requires_grad = False

    gen.to(device)

    gen.eval()

    return gen


def infer_from_silence(file: str = "generated\gan_cnn_9_12"):
    set_seed(42)

    gen = load_model(f"{file}.pth")

    with torch.no_grad():
        z = torch.randn(5, 100).to(device)
        fake = gen(z, torch.zeros(5, 64, 88).to(device))

        fake_pr = [matrix2pianoroll(track.cpu().numpy().T) for track in fake]

        for i in range(5):
            ax = plt.figure(figsize=(8, 3)).add_subplot(111)
            ppr.plot_pianoroll(ax, fake_pr[i], xtick="auto")

        plt.title("Generated from silence")
        plt.show()


if __name__ == "__main__":
    infer_from_silence("checkpoints\gan_cnn_12_18")
    show("generated\gan_cnn_12_18")
