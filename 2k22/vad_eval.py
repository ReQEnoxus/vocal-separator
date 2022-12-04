import librosa
import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write
import time

BINARIZATION_COEFF = 0.215
VOCAL_SENSITIVITY = 0.64
SMOOTH_RANGE = 24


class Track(object):

    def __init__(self, track, part_size=9):
        self.stft = librosa.stft(librosa.to_mono(track))
        padding = np.zeros((self.stft.shape[0], part_size // 2))
        self.stftPadded = np.abs(
            np.concatenate((padding, librosa.stft(librosa.to_mono(track)), padding, padding), axis=1))
        self.size = part_size

    def __iter__(self):
        return Track.TrackIterator(self.stftPadded, self.size)

    class TrackIterator:

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


def binarize(a, coeff=BINARIZATION_COEFF):
    if a > coeff:
        return 1
    else:
        return 0


test_song = librosa.load("sample2.wav", sr=22050)[0]

track = Track(test_song)

sgdOpt = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model_vad = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(1025, 9, 1)),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_vad.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=sgdOpt, metrics=['accuracy'])
model_vad.load_weights("artifacts/b66d5b6e-7bb0-4b7a-977d-de503ddf3452/weights-0.85.hdf5")

print("model loaded")

bin_mask = []

print("processing started")
start_time = time.time()
smooth_iter = 0
previous_vox = False
for stft in track:
    stft = np.expand_dims(stft, axis=2)
    stft = np.abs(tf.expand_dims(stft, 0))
    pred_bin = []
    if previous_vox:
        for i in range(stft.shape[1]):
            pred_bin.append(1)
        smooth_iter += 1
        if smooth_iter == SMOOTH_RANGE:
            previous_vox = False
    else:
        vocal_prediction = model_vad.predict(stft)[0]
        if vocal_prediction > VOCAL_SENSITIVITY:
            for i in range(stft.shape[1]):
                pred_bin.append(1)
            previous_vox = True
            smooth_iter = 0
        else:
            for i in range(stft.shape[1]):
                pred_bin.append(0)

    bin_mask.append(pred_bin)

binary = np.array(bin_mask).T

binary = np.delete(binary, np.s_[track.stft.shape[1]:binary.shape[1]], 1)

vox_stft = track.stft * binary

print("finished processing in %s seconds" % (time.time() - start_time))

write("after_2_smooth.wav", 22050, librosa.istft(vox_stft))
