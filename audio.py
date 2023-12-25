"""Audio file feature extractor and visualization tools"""
import os

import librosa
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


class Audio:
    """Extracts features from an audio file

    Parameters
    ----------
    filename: str | os.PathLike
        The audio file to use, can be of any file format given that the codecs
        are properly installed.

    kwargs
        All other keyword arguments are passed to the librosa.load such as slicing.
    """

    def __init__(self, filename: str | os.PathLike, **kwargs):
        self.y, self.sr = librosa.load(filename, **kwargs)
        self.time = np.arange(len(self.y)) / self.sr
        self.tempo, self.beat_frames, self.beat_times = (None,) * 3
        self.melspectrogram = None
        self.stft = None
        self.extract_beat()

    def extract_stft(self, **kwargs) -> np.ndarray:
        """Extracts Short-time-Fourier-transform using a wavelet with overlapping sliding windows.
        The instance variable returned self.stft is a complex ndarray which is the raw STFT for the
        given track.

        Parameters
        ----------
        kwargs
            All keyword arguments are passed to librosa.stft method.
        """
        self.stft = librosa.stft(y=self.y, **kwargs)

    def extract_beat(self, **kwargs) -> None:
        """Gets the beat of a track. This method does not return anything, rather sets instance
        variables of self.tempo, self.beat_frames and self.beat_times.

        Parameters
        ----------
        kwargs
            All keyword arguments are passed to the librosa.beat.beat_track method.
        """
        self.tempo, self.beat_frames = librosa.beat.beat_track(
            y=self.y, sr=self.sr, **kwargs
        )
        self.beat_times = librosa.frames_to_time(frames=self.beat_frames, sr=self.sr)

    def get_mel_spectrogram(self, **kwargs) -> np.ndarray:
        """Creates a melodic or (MEL) spectrogram which is useful for plotting frequency vs time
        that is usaually used to visualize the frequency variations over the track's time period.

        Parameters
        ----------
        kwargs
            All keyword arguments are passed to the librosa.features.melspectrogram

        Returns
        -------
        np.ndarray
            The numpy array of spetrograms with the final dim represents the time domain.
            Typically a matrix of spectrogram vs time.
        """
        self.melspectrogram = librosa.feature.melspectrogram(
            y=self.y,
            sr=self.sr,
            n_mels=kwargs.pop("n_mels", 128),
            fmax=kwargs.pop("fmax", 8000),
            **kwargs
        )
        return self.melspectrogram

    def plot_beats(self) -> Axes:
        """Creates a plot of beats vs time for the given track."""
        if self.beat_times is None:
            raise ValueError("extract_beat method needs to be run to obtain the plots.")
        ax = plt.subplot()
        ax.plot(librosa.times_like(self.y, sr=self.sr), self.y)
        ax.vlines(self.beat_times, -1, 1, color="r", alpha=0.5)
        ax.set_ylabel("Amplitude")
        ax.set_title("Detected Beats")
        plt.tight_layout()
        plt.show()
        return ax

    @staticmethod
    def plot_mel_spectrogram(spectrogram: np.ndarray):
        """Creates a plot of the melodic (MEL) spectrogram for visualizing the frequency vs time.

        Parameters
        ----------
        spectrogram: np.ndarray
            The spectrogram for the given track, must be in"""
        # spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        plt.figure(figsize=(10, 4))
        fig, ax = plt.subplots()
        librosa.display.specshow(
            librosa.power_to_db(spectrogram, ref=np.max),
            y_axis="mel",
            fmax=8000,
            x_axis="time",
            ax=ax,
        )
        fig.colorbar(format="%+2.0f dB")
        ax.set_title("Mel spectrogram")
        plt.tight_layout()
        plt.show()
        return ax
