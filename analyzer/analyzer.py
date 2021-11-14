from model import models
from entity import entities
from scipy.io.wavfile import write
import librosa
import numpy as np
import tensorflow as tf


class Analyzer(object):
    def __init__(self, model: models.Model, verbose=True):
        self.model = model
        self.verbose_output=verbose

    def binarize(self, a, binarization_coeff):
        if a > binarization_coeff:
            return 1
        else:
            return 0

    def extract_vocals(self, input_path, output_path, binarization_coeff=0.215):
        song = entities.Song(librosa.load(input_path, sr=44100)[0])
        bin_mask = []

        for stft in song:
            stft = np.expand_dims(stft, axis=2)
            stft = np.abs(tf.expand_dims(stft, 0))
            prediction = self.model.predict(stft)[0]
            pred_bin = []
            for p in prediction:
                pred_bin.append(self.binarize(p, binarization_coeff))
            bin_mask.append(pred_bin)

        binary = np.array(bin_mask).T
        binary = np.delete(binary, np.s_[song.stft.shape[1]:binary.shape[1]], 1)
        vox_stft = song.stft * binary
        write(output_path, 44100, librosa.istft(vox_stft))
