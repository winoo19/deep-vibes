import os
import numpy as np
import pypianoroll as ppr
from midi2audio import FluidSynth

# own libraries
from src.data import RESOLUTION

OUTPUT_PATH = "output"
SOUNDFONTS_PATH = "soundfonts"
SOUND_FONT = "Yamaha C5 Grand-v2.4.sf2"


def pianoroll2audio(pianoroll: np.ndarray, output_path: str) -> None:
    """
    This function converts a pianoroll to an audio file and saves
    it to the sounds folder.

    Args:
        pianoroll: pianoroll to convert.
        output_path: path to save the audio file.
    """

    if pianoroll.shape[1] == 88:
        # Pad the pianoroll to 128 keys
        pianoroll = np.pad(pianoroll, ((0, 0), (21, 19)))
    elif pianoroll.shape[1] != 128:
        raise ValueError("Invalid number of keys.")

    os.makedirs(SOUNDFONTS_PATH, exist_ok=True)

    output_path = os.path.join(OUTPUT_PATH, output_path)

    fs = FluidSynth(os.path.join(SOUNDFONTS_PATH, SOUND_FONT))
    fs.midi_to_audio(pianoroll, output_path)

    return None


def pianoroll2midi(pianoroll: np.ndarray, output_path: str) -> None:
    """
    This function converts a pianoroll to a midi file and saves
    it to the sounds folder.

    Args:
        pianoroll: pianoroll to convert.
        output_path: path to save the midi file.
    """
    if pianoroll.shape[1] == 88:
        # Pad the pianoroll to 128 keys
        pianoroll = np.pad(pianoroll, ((0, 0), (21, 19)))
    elif pianoroll.shape[1] != 128:
        raise ValueError("Invalid number of keys.")

    os.makedirs(SOUNDFONTS_PATH, exist_ok=True)

    output_path = os.path.join(OUTPUT_PATH, output_path)

    track = ppr.StandardTrack(pianoroll=pianoroll)

    multitrack = ppr.Multitrack(tracks=[track], tempo=120, resolution=RESOLUTION)

    multitrack.write(output_path)
