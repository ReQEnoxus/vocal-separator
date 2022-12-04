import numpy as np
import librosa
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import uuid
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa

AUTOTUNE = tf.data.experimental.AUTOTUNE

TRAIN_NUM = 0
VAL_NUM = 0
TEST_NUM = 0
BATCH_SIZE = 256

def generate_data(df: pd.DataFrame, batch_size=256):
    """
    :param df: dataframe that contains paths to tracks and to corresponding masks
    :param batch_size: the size of batch
    :return: yields the batch of features and labels
    """

    df = df.sample(frac=1)
    offset = 0
    limit = batch_size
    while True:
        batch_feature = []
        batch_label = []
        batch_df = df.iloc[offset:limit]  # берем слайс датафрейма, из которого формируем батч

        batch_df = batch_df.sample(frac=1)  # перемешиваем

        for index, row in batch_df.iterrows():
            track = librosa.load(row['track'], sr=22050)[0]  # загружаем трек
            stft = np.abs(librosa.stft(track, n_fft=2048))  # вычисляем оконное преобразование и поэлементно находим модуль
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


def main():
    train_id = str(uuid.uuid4())
    artifact_dir = f"main_artifacts/{train_id}"
    os.mkdir(artifact_dir)
    dataframe_train = pd.read_csv("main_meta_train.csv", names=['track', 'mask'], header=None)
    dataframe_val = pd.read_csv("main_meta_val.csv", names=['track', 'mask'], header=None)

    model = tf.keras.models.Sequential([
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
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1025, activation='sigmoid')
    ])
    # model = tf.keras.models.load_model("main_artifacts/e92a4d68-0894-4c9f-b9b8-cfd476567f44/vad-model-e92a4d68-0894-4c9f-b9b8-cfd476567f44.h5")

    TRAIN_NUM = int(dataframe_train.shape[0])
    VAL_NUM = int(dataframe_val.shape[0])

    opt = tf.keras.optimizers.SGD(lr=0.02, decay=1e-6, momentum=0.87, nesterov=True)
    # opt = tfa.optimizers.NovoGrad(lr=0.01)
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    # model.compile(loss=tfa.losses.SigmoidFocalCrossEntropy(), optimizer=sgdOpt, metrics=['accuracy'])
    model.load_weights("main_artifacts/21c0f28e-a0b5-4f8b-9524-170710f8ff14/weights-0.66.hdf5")
    global dataset_train
    dataset_train = tf.data.Dataset.from_generator(lambda: generate_data(dataframe_train, BATCH_SIZE),
                                                   output_types=(tf.float32, tf.float64),
                                                   output_shapes=((None, 1025, 9, 1), (None, 1025)))
    dataset_val = tf.data.Dataset.from_generator(lambda: generate_data(dataframe_val, BATCH_SIZE),
                                                 output_types=(tf.float32, tf.float64),
                                                 output_shapes=((None, 1025, 9, 1), (None, 1025)))

    filepath = artifact_dir + "/weights-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = model.fit(dataset_train,
                        epochs=20,
                        steps_per_epoch=TRAIN_NUM // BATCH_SIZE,
                        validation_data=dataset_val,
                        validation_steps=VAL_NUM // BATCH_SIZE,
                        callbacks=callbacks_list,
                        initial_epoch=0)

    model.save(f'{artifact_dir}/vad-model-{train_id}.h5')  # saves the model

    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f"{artifact_dir}/plot-{train_id}.png", transparent=True)
    plt.show()


if __name__ == '__main__':
    main()
