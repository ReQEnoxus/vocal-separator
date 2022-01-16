import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint


class MainVocalModel(object):
    model: tf.keras.models.Model

    def __init__(self, weights=None):
        self.setup_model()

        if weights is not None:
            self.model.load_weights(weights)

    def setup_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(1025, 25, 1)),
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
            tf.keras.layers.Dense(1025, activation='sigmoid')
        ])

        sgdOpt = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=sgdOpt, metrics=['accuracy'])

    def fit(
            self,
            dataset_train,
            epochs,
            steps_per_epoch,
            validation_data,
            validation_steps,
            initial_epoch,
            result_name
    ):
        filepath = "weights/main/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        history = self.model.fit(dataset_train,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=validation_data,
                                 validation_steps=validation_steps,
                                 callbacks=callbacks_list,
                                 initial_epoch=initial_epoch)

        self.model.save(f'{result_name}.h5')

        training_loss = history.history['loss']
        test_loss = history.history['val_loss']

        return training_loss, test_loss

    def predict(self, stft_chunk):
        return self.model.predict(stft_chunk)


class VADModel(object):
    model: tf.keras.models.Model

    def __init__(self, weights=None):
        self.setup_model()

        if weights is not None:
            self.model.load_weights(weights)

    def setup_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(1025, 25, 1)),
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

        sgdOpt = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=sgdOpt, metrics=['accuracy'])

    def fit(
            self,
            dataset_train,
            epochs,
            steps_per_epoch,
            validation_data,
            validation_steps,
            initial_epoch,
            result_name
    ):
        filepath = "weights/vad/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        history = self.model.fit(dataset_train,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_data=validation_data,
                                 validation_steps=validation_steps,
                                 callbacks=callbacks_list,
                                 initial_epoch=initial_epoch)

        self.model.save(f'{result_name}.h5')

        training_loss = history.history['loss']
        test_loss = history.history['val_loss']

        return training_loss, test_loss

    def predict(self, stft_chunk):
        return self.model.predict(stft_chunk)
