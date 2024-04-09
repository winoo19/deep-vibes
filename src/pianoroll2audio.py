import os
import numpy as np
import pypianoroll as ppr
from midi2audio import FluidSynth

# own libraries
from src.data import RESOLUTION

OUTPUT_FOLDER = "output"
SOUNDFONTS_FOLDER = "soundfonts"
SOUND_FONT = "Yamaha C5 Grand-v2.4.sf2"


def pianoroll2audio(pianoroll: np.ndarray, filename: str) -> None:
    """
    This function converts a pianoroll to an audio file and saves
    it to the sounds folder.

    Args:
        pianoroll: pianoroll to convert.
        filename: name of the audio file.
    """

    os.makedirs(SOUNDFONTS_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Convert pianoroll to midi
    pianoroll2midi(pianoroll, filename)

    output_path = os.path.join(OUTPUT_FOLDER, f"{filename}.wav")

    fs = FluidSynth(os.path.join(SOUNDFONTS_FOLDER, SOUND_FONT))
    fs.midi_to_audio(os.path.join(OUTPUT_FOLDER, f"{filename}.mid"), output_path)

    return None


def pianoroll2midi(pianoroll: np.ndarray, filename: str) -> None:
    """
    This function converts a pianoroll to a midi file and saves
    it to the sounds folder.

    Args:
        pianoroll: pianoroll to convert.
        filename: name of the midi file.
    """

    if pianoroll.shape[1] == 88:
        # Pad the pianoroll to 128 keys
        pianoroll = np.pad(pianoroll, ((0, 0), (21, 19)))
    elif pianoroll.shape[1] != 128:
        raise ValueError("Invalid number of keys.")

    mid = ppr.Multitrack(
        tracks=[
            ppr.StandardTrack(
                name="piano",
                program=0,
                is_drum=False,
                pianoroll=pianoroll,
            )
        ],
        tempo=np.ones(pianoroll.shape[0]) * 120,
        resolution=RESOLUTION,
    )

    mid.write(os.path.join(OUTPUT_FOLDER, f"{filename}.mid"))
