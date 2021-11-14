import librosa
import numpy as np


class Song(object):

    def __init__(self, track, part_size=25):
        self.stft = librosa.stft(librosa.to_mono(track))
        padding = np.zeros((self.stft.shape[0], part_size // 2))
        self.stftPadded = np.abs(
            np.concatenate((padding, librosa.stft(librosa.to_mono(track)), padding, padding), axis=1))
        self.size = part_size

    def __iter__(self):
        return Song.SongIterator(self.stftPadded, self.size)

    class SongIterator:

        def __init__(self, stft, part_size):
            self.stft = stft
            self.index = 0
            self.step = part_size
            self.end_index = stft.shape[1]

        def __iter__(self):
            return self

        def __next__(self):
            if self.index + self.step < self.end_index:
                start_index = self.index
                end_index = self.index + self.step
                self.index += 1
                return self.stft[:, start_index: end_index]
            else:
                raise StopIteration
            