from model import models
from entity import entities
from scipy.io.wavfile import write
import librosa
import numpy as np
import tensorflow as tf


class Analyzer(object):
    def __init__(self, model: models.MainVocalModel, vad_model: models.VADModel, verbose=True):
        self.model = model
        self.vad_model = vad_model
        self.verbose_output=verbose

    def binarize(self, a, binarization_coeff):
        if a > binarization_coeff:
            return 1
        else:
            return 0

    def extract_vocals(self, input_path, output_path, binarization_coeff=0.215, smooth_coeff=30, vocal_sensitivity=0.9):
        song = entities.Song(librosa.load(input_path, sr=44100)[0])
        bin_mask = []

        smooth_iter = 0
        previous_vox = False
        for stft in song:
            stft = np.expand_dims(stft, axis=2)
            stft = np.abs(tf.expand_dims(stft, 0))
            vocal_prediction = self.vad_model.predict(stft)[0]
            pred_bin = []
            if previous_vox:
                prediction = self.model.predict(stft)[0]
                for p in prediction:
                    pred_bin.append(self.binarize(p, binarization_coeff))
                smooth_iter += 1
                if smooth_iter == smooth_coeff:
                    previous_vox = False
            elif vocal_prediction > vocal_sensitivity:
                prediction = self.model.predict(stft)[0]
                for p in prediction:
                    pred_bin.append(self.binarize(p, binarization_coeff))
                previous_vox = True
                smooth_iter = 0
            else:
                for i in range(stft.shape[1]):
                    pred_bin.append(0)

            bin_mask.append(pred_bin)

        binary = np.array(bin_mask).T
        binary = np.delete(binary, np.s_[song.stft.shape[1]:binary.shape[1]], 1)
        vox_stft = song.stft * binary
        write(output_path, 44100, librosa.istft(vox_stft))
