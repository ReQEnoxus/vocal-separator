from math import ceil
import os
import librosa
from scipy.io import wavfile
# import musdb
import numpy as np
from skimage import io
from scipy.io.wavfile import write


def divide_audio_into_frames(voice, noise, full, frames=9) -> None:
    """
    Divide audio track into equal pieces and save them
    :param song: comes as MultiTrack representation from MUSDB
    :param frames: parts to divide track by
    """
    assert voice is not None
    assert noise is not None
    vocals_stft = librosa.stft(voice)
    accompaniment_stft = librosa.stft(noise)
    song_stft = librosa.stft(full)
    parts = ceil(song_stft.shape[1] / frames)
    for i in range(parts - 1):
        acc_matrix = accompaniment_stft[:, i * frames: i * frames + frames]
        vox_matrix = vocals_stft[:, i * frames: i * frames + frames]
        print(f"!! {vox_matrix.shape}")
        if vox_matrix.shape[1] < frames or acc_matrix.shape[1] < frames:
            return
        acc_mid = acc_matrix[:, ceil(frames / 2) - 1]
        vox_mid = vox_matrix[:, ceil(frames / 2) - 1]

        mid_mask = binary_mask(acc_mid, vox_mid)

        if np.count_nonzero(mid_mask) >= 180:
            track_part = song_stft[:, i * frames: i * frames + frames]
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


def save_track_and_its_mask(track, mask, directory_to_save='E:\\Projects\\vocal-separator\\2k22\\dataset_main_test'):
    """
    Track will be saved as wav file, mask will be saved in .npy format (which is easy to work with using np.load(filename))
    :param track: track to save, should be ndarray data type
    :param mask: computed mask, should by ndarray data type
    """

    # track = librosa.istft(track)
    global TRACK_NAME
    tracks_directory = os.path.join(directory_to_save, "tracks")
    masks_directory = os.path.join(directory_to_save, "masks")
    if not os.path.exists(tracks_directory):
        os.mkdir(tracks_directory)

    if not os.path.exists(masks_directory):
        os.mkdir(masks_directory)

    # print('Saving ' + str(TRACK_NAME) + 'track_file')
    # spectrogram_image(track, 44100, directory_to_save + '/img/' + str(TRACK_NAME), 512, 1025)
    write(directory_to_save + '\\tracks\\' + str(TRACK_NAME) + '.wav', 22050, librosa.istft(track))
    np.save(directory_to_save + '\\masks\\' + str(TRACK_NAME), mask)
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

# db = musdb.DB(root='E:/AEar/datasets/musdb18')
# for i in range(db.__len__()):
#     divide_audio_into_frames(db[i])
#     print(i, "track finished")

noise_root_path = 'E:\\Projects\\vocal-separator\\2k22\\raw_data\\test\\Noise_testing'
voice_root_path = 'E:\\Projects\\vocal-separator\\2k22\\raw_data\\test\\CleanSpeech_testing'
full_root_path = 'E:\\Projects\\vocal-separator\\2k22\\raw_data\\test\\NoisySpeech_testing'
noise_count = 0
voice_count = 0
# Iterate directory
for path in os.listdir(noise_root_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(noise_root_path, path)):
        noise_count += 1
print('noise file count:', noise_count)

for path in os.listdir(voice_root_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(voice_root_path, path)):
        voice_count += 1
print('voice file count:', voice_count)

for i in range(0, noise_count):
    file_name = f"{i}.wav"
    noise_file_path = os.path.join(noise_root_path, file_name)
    noise_data, sample_rate = librosa.load(noise_file_path, sr=22050)

    voiсe_file_path = os.path.join(voice_root_path, file_name)
    voice_data, sample_rate = librosa.load(voiсe_file_path, sr=22050)

    full_file_path = os.path.join(full_root_path, file_name)
    full_data, sample_rate = librosa.load(full_file_path, sr=22050)

    divide_audio_into_frames(voice_data, noise_data, full_data)
    print(i, "track finished")