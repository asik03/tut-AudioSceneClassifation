from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import adam

import matplotlib.pyplot as plt


class SceneNetwork:
    def __init__(self):
        self.model = None

    def build_model(self, input_shape):
        self.model = Sequential()
        self.conv_block(32, input_shape)
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.conv_block(64)
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.conv_block(128)
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.conv_block(256)
        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dense(15, activation='softmax'))

        optimizer = adam()

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])
        self.model.summary()
        # We do not concatenate. Is it necessary?
        # self.model.add(Dense(1024, activation='relu'))
        # self.model.add(Dense(15, activation='softmax'))

    def conv_block(self, n_filters, input_shape=None):
        if input_shape is not None:
            self.model.add(BatchNormalization(input_shape=input_shape))
        else:
            self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(n_filters, (3, 3), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(n_filters, (3, 3), padding='same'))

    def train_model(self, x_train, y_train, x_val, y_val, batch_size=32, epochs=50):
        history = self.model.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 shuffle=True,
                                 validation_data=(x_val, y_val))
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def evaluate_model(self, x_eval, y_eval):
        scores = self.model.evaluate(x_eval, y_eval, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def train_model_generator(self, train_generator, validation_generator=None, epochs=50):
        self.model.fit_generator(generator=train_generator,
                                 validation_data=validation_generator,
                                 epochs=epochs)

    def evaluate_model_generator(self, evaluate_generator):
        scores = self.model.evaluate_generator(generator=evaluate_generator,
                                               verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def save_model(self, model_path):
        self.model.save(model_path)
