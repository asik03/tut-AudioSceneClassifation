from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D


class SceneNetwork:
    def __init__(self):
        self.model = None

    def build_model(self, input_shape):
        self.model = Sequential()
        self.conv_block(32, input_shape)
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.conv_block(64)
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.conv_block(128)
        self.model.add(MaxPooling2D(pool_size=(3, 3)))
        self.conv_block(256)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(15, activation='softmax'))
        # We do not concatenate. Is it necessary?
        # self.model.add(Dense(1024, activation='relu'))
        # self.model.add(Dense(15, activation='softmax'))

    def conv_block(self, n_filters, input_shape=None):
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        if input_shape is not None:
            self.model.add(Conv2D(n_filters, (3, 3), padding='same', input_shape=input_shape))
        else:
            self.model.add(Conv2D(n_filters, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(n_filters, (3, 3), padding='same'))

    def train_model(self, train_generator, validation_generator=None, epochs=50):
        self.model.fit_generator(generator=train_generator,
                                 validation_generator=validation_generator,
                                 epochs=epochs)

    def evaluate_model(self, evaluate_generator):
        scores = self.model.evaluate_generator(generator=evaluate_generator,
                                               verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def save_model(self, model_path):
        self.model.save(model_path)
