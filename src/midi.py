import os
from matplotlib import pyplot as plt
import numpy as np
from midi2audio import FluidSynth
import pretty_midi

OUTPUT_FOLDER = "out"
SOUNDFONTS_FOLDER = "soundfonts"
SOUND_FONT = "Studio_Grand.sf2"


def midi2pianoroll(midi_file: str, fs: int) -> np.ndarray:
    """
    This function converts a MIDI file to a pretty_midi pianoroll
    with shape (128, frames).

    Args:
        midi_file: path to the MIDI file.
        fs: sampling frequency.

    Returns:
        piano-roll: the piano-roll of the MIDI file.
    """

    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)

    # Get the piano-roll
    piano_roll = midi_data.get_piano_roll(fs=fs)

    return piano_roll


# Colin Raffel (2018) pretty-midi/examples/reverse_pianoroll.py [Source code]: https://github.com/craffel/pretty-midi/blob/main/examples/reverse_pianoroll.py
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    """Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    """
    notes, _ = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], "constant")

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time,
            )
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


def pianoroll2midi(pianoroll: np.ndarray, fs: int, save_path: str) -> None:
    """
    This function converts a pretty_midi pianoroll to a MIDI file.
    """
    pm = piano_roll_to_pretty_midi(pianoroll, fs=fs, program=0)
    pm.write(save_path)


def pianoroll2audio(pianoroll: np.ndarray, fs: int, filename: str) -> None:
    """
    This function converts a pretty_midi pianoroll to a MIDI file and then
    to an audio file.
    """
    pianoroll2midi(pianoroll, fs=fs, save_path=f"{OUTPUT_FOLDER}/{filename}.mid")

    # Convert the MIDI file to an audio file
    FluidSynth(f"{SOUNDFONTS_FOLDER}/{SOUND_FONT}").midi_to_audio(
        f"{OUTPUT_FOLDER}/{filename}.mid", f"{OUTPUT_FOLDER}/{filename}.wav"
    )


def pianoroll2matrix(pianoroll: np.ndarray) -> np.ndarray:
    """
    This function parses a pretty_midi pianoroll to the our defined standard form to train the models.
    - Velcoity values are normalized to [0, 1]
    - Pitch channel is cropped to 88 keys.
    """

    # Crop the piano-roll to the piano keys and transpose it
    piano_roll_cropped = pianoroll[21:109].T

    # Normalize the velocity values
    piano_roll_clipped = np.clip(piano_roll_cropped, 0, 128)
    piano_roll_normalized = piano_roll_clipped / 128

    return piano_roll_normalized.astype(np.float32)


def matrix2pianoroll(matrix: np.ndarray) -> np.ndarray:
    """
    This function convert our matrix representation of pianoroll to the pretty_midi form.
    - Velocity values are scaled back to [0, 127]
    - Pitch channel is padded to 128 keys.

    Args:
        matrix: the piano-roll matrix. Shape: (frames, 88)

    Returns:
        piano-roll: the piano-roll of the MIDI file. Shape: (128, frames)
    """

    # Scale the velocity values and transpose the matrix
    piano_roll_scaled_T = (matrix * 127).astype(np.uint8).T

    # Pad the piano-roll to 128 keys
    piano_roll_padded = np.pad(piano_roll_scaled_T, ((21, 19), (0, 0)), "constant")

    return piano_roll_padded


def trim_silence(matrix: np.ndarray) -> np.ndarray:
    """
    This function trims the silence from the beginning and the end of the piano-roll.

    Args:
        matrix: the piano-roll matrix. Shape: (frames, 88)
    """

    # Get the indices of the first and last notes
    collapsed = matrix.sum(axis=1).nonzero()[0]
    start = collapsed[0]
    end = collapsed[-1]

    return matrix[start:end]


def generate_random_pianoroll(n_notes: int):
    """
    Generate a random pianoroll of n_notes notes.
    """
    matrix = np.zeros((n_notes, 88), dtype=np.float32)

    t = 0
    while t < n_notes:
        # Random number of time steps from 1 to 16
        time_steps = np.random.randint(1, 17)
        # Random number of notes from 0 to 10
        n = np.random.randint(0, 11)
        # Random notes
        notes = np.random.randint(0, 88, n)

        matrix[t : t + time_steps, notes] = 1

        t += time_steps

    return matrix


def generate_random_matrix(n_notes: int):
    """ """

    MAX_N_PITCHES = 5
    PROB_START_PLAYING = 0.65
    PROB_ADD_SOME_NOTES = 0.2
    PROB_REMOVE_SOME_NOTES = 0.2
    PROB_STOP_PLAYING_ALL_NOTES = 0.05

    MIN_PITCH = 10
    MAX_PITCH = 70

    matrix = np.zeros((n_notes, 88), dtype=np.float32)

    n_pitches = np.random.randint(1, MAX_N_PITCHES + 1)
    notes = np.random.randint(MIN_PITCH, MAX_PITCH, n_pitches)

    playing = False

    for t in range(n_notes):
        # Actions: play the notes, stop playing some notes, stop playing all notes

        if playing:
            matrix[t, notes] = 0.95

        # Start play the notes
        if playing == False and np.random.rand() < PROB_START_PLAYING:
            n_pitches = np.random.randint(1, MAX_N_PITCHES + 1)
            notes = np.random.randint(MIN_PITCH, MAX_PITCH, n_pitches)
            matrix[t, notes] = 0.95
            playing = True

        # Add some note
        if playing == True and np.random.rand() < PROB_ADD_SOME_NOTES:
            n_pitches = np.random.randint(1, (MAX_N_PITCHES + 1) // 2)
            notes = np.unique(
                np.concatenate(
                    [notes, np.random.randint(MIN_PITCH, MAX_PITCH, n_pitches)]
                )
            )

        # Stop playing some notes
        if playing == True and np.random.rand() < PROB_REMOVE_SOME_NOTES:
            if len(notes) == 1:
                notes_to_stop = notes
                notes = None
                playing = False
            else:
                notes_to_stop = np.random.choice(
                    notes, np.random.randint(1, len(notes)), replace=False
                )
            matrix[t, notes_to_stop] = 0

        # Stop playing all notes
        if playing == True and np.random.rand() < PROB_STOP_PLAYING_ALL_NOTES:
            matrix[t, notes] = 0
            notes = None
            playing = False

    return matrix


def generate_noise(n_notes: int = 16 * 5):
    """
    Generate a random midi file with n_notes notes.
    """

    matrix = (np.random.rand(n_notes, 88) > 0.8).astype(np.float32)
    pianoroll = matrix2pianoroll(matrix)
    pianoroll2midi(pianoroll, 16, "out/noise_midi.mid")

    # Save image
    plt.imshow(matrix.T, aspect="auto", cmap="gray")
    plt.title("Noise pianoroll")
    plt.xlabel("Time")
    plt.ylabel("Pitch")
    plt.savefig("out/noise_pianoroll.png")


if __name__ == "__main__":
    # print(os.listdir("data/midi")[247])
    # npy_path = os.path.join("data", "npy")
    # matrix = np.load(os.path.join(npy_path, "pianoroll_247.npy"))

    # print("Pianoroll shape:", matrix.shape)

    # pianoroll = matrix2pianoroll(matrix)
    # print("Pianoroll shape:", pianoroll.shape)

    # pianoroll2audio(pianoroll, fs=16, filename="pianoroll_247")

    generate_noise()
