from math import ceil
import os
import librosa
import musdb
import numpy as np
from skimage import io
from scipy.io.wavfile import write


def divide_audio_into_frames(song, step=25, frames=25) -> None:
    """
    Divide audio track into equal pieces and save them
    :param song: comes as MultiTrack representation from MUSDB
    :param frames: parts to divide track by
    :param step: stride of division window
    """
    assert song is not None
    vocals_stft = librosa.stft(librosa.to_mono(song.targets['vocals'].audio.T))
    accompaniment_stft = librosa.stft(librosa.to_mono(song.targets['accompaniment'].audio.T))
    song_stft = librosa.stft(librosa.to_mono(song.audio.T))
    parts = ceil(song_stft.shape[1] / step)
    for i in range(parts - 1):
        acc_matrix = accompaniment_stft[:, i * step: i * step + frames]
        vox_matrix = vocals_stft[:, i * step: i * step + frames]

        acc_mid = acc_matrix[:, ceil(frames / 2) - 1]
        vox_mid = vox_matrix[:, ceil(frames / 2) - 1]

        mid_mask = binary_mask(acc_mid, vox_mid)

        if np.count_nonzero(mid_mask) > 150:
            track_part = song_stft[:, i * step: i * step + frames]
            save_track_and_its_mask(track_part, mid_mask)
    return


def binary_mask(source, target):
    """
        :param source: Everything in the mix except for target (for example: for vocals it would be accompaniment)
        :param target: Target source that needs to be separated
        :return: Binary mask for separation the target from full mix
        """
    assert source is not None and target is not None
    assert source.shape == target.shape

    mask = np.zeros(source.shape)

    iterator = np.nditer(source, flags=['multi_index'])

    while not iterator.finished:
        source_item = iterator[0]
        target_item = target[iterator.multi_index]

        if np.abs(target_item) > np.abs(source_item):
            mask[iterator.multi_index] = 1

        iterator.iternext()

    return mask


def save_track_and_its_mask(track, mask, directory_to_save='E:/AEar/datasets/dataset_25_full_VOXOnly_step12'):
    """
    Track will be saved as wav file, mask will be saved in .npy format (which is easy to work with using np.load(filename))
    :param track: track to save, should be ndarray data type
    :param mask: computed mask, should by ndarray data type
    """

    # track = librosa.istft(track)
    global TRACK_NAME
    if not os.path.exists(directory_to_save):
        os.mkdir(directory_to_save)
        os.mkdir(directory_to_save + '/tracks')
        os.mkdir(directory_to_save + '/masks')
    # print('Saving ' + str(TRACK_NAME) + 'track_file')
    # spectrogram_image(track, 44100, directory_to_save + '/img/' + str(TRACK_NAME), 512, 1025)
    write(directory_to_save + '/tracks/' + str(TRACK_NAME) + '.wav', 44100, librosa.istft(track))
    np.save(directory_to_save + '/masks/' + str(TRACK_NAME), mask)
    TRACK_NAME += 1


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def spectrogram_image(y, sr, out, hop_length, n_mels):
    # use log-melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=2048, hop_length=hop_length)
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)

    # mels = numpy.abs(librosa.stft(y))

    # min-max scale to fit inside 8-bit range
    img = scale_minmax(mels, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy

    # save as PNG
    # numpy.save(out, img)
    io.imsave(out + ".png", img)


TRACK_NAME = 0

db = musdb.DB(root='E:/AEar/datasets/musdb18')
for i in range(db.__len__()):
    divide_audio_into_frames(db[i], step=12)
    print(i, "track finished")
