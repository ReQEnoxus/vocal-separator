import numpy as np
import librosa
import pandas as pd


class DataGenerator(object):
    """
    Содержит генераторную функцию, возвращающую датасет по путям, записанным в датафрейм. Датафрейм должен содержать
    две колонки с лейблами 'track' и 'mask', в первой колонке должны идти пути к аудиофайлам в формате wav, а во второй
    соответствующие им лейблы в формате npy
    """
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def generate_data(self, batch_size=256):
        df = self.dataframe.sample(frac=1)
        offset = 0
        limit = batch_size
        while True:
            batch_feature = []
            batch_label = []
            batch_df = df.iloc[offset:limit]  # берем слайс датафрейма, из которого формируем батч

            batch_df = batch_df.sample(frac=1)  # перемешиваем

            for index, row in batch_df.iterrows():
                track = librosa.load(row['track'], sr=44100)[0]  # загружаем трек
                stft = np.abs(librosa.stft(track))  # вычисляем оконное преобразование и поэлементно находим модуль
                stft = np.expand_dims(stft, axis=2)  # добавляем дополнительное измерение
                mask = np.load(row['mask'])
                batch_feature.append(stft)
                batch_label.append(mask)

            offset = limit
            limit += batch_size

            if limit > int(
                    df.shape[0]):  # если прошлись до конца, то сбрасываем лимит и оффсет и снова перемешиваем датафрейм
                offset = 0
                limit = batch_size
                df = df.sample(frac=1)

            yield np.array(batch_feature), np.array(batch_label)
