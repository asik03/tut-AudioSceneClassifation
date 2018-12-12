import os
import numpy as np
import keras
import librosa


class FeaturesGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data_dir, labels, batch_size=32, n_classes=15, shuffle=True, method='mfcc', n_features=20):
        """Initialization"""
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.file_names = self.__get_file_names()
        self.labels = labels
        self.indexes = self.__get_indexes()
        self.method = method
        self.n_features = n_features

    def __get_file_names(self):
        file_names = os.listdir(self.data_dir)
        file_names = list(filter(lambda x: x.ends_with('.wav'), file_names))

        return file_names

    def __get_indexes(self):
        indexes = np.arange(len(self.file_names))
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of file names
        file_names_temp = [self.file_names[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(file_names_temp)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = self.__get_indexes()

    def __data_generation(self, file_names_temp):
        """Generates data containing batch_size samples"""

        list_x = []
        list_y = []
        for i, file_name in enumerate(file_names_temp):
            # It will convert to mono
            data, fs = librosa.load(os.path.join(self.data_dir, file_name), sr=None)

            if self.method == 'mfcc':
                x_i = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=self.n_features)
            elif self.method == 'chroma_cqt':
                x_i = librosa.feature.chroma_cqt(y=data, sr=fs, n_chroma=self.n_features)
            elif self.method == 'both':
                mfcc = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=self.n_features)
                chroma = librosa.feature.chroma_cqt(y=data, sr=fs, n_chroma=self.n_features)
                x_i = np.concatenate((mfcc, chroma), axis=1)
            else:
                raise Exception('Method not recognized')

            class_name = file_name.split('-')[-1].split('.')[0]

            y_i = self.labels.index(class_name)

            list_x.append(x_i)
            list_y.append(y_i)

        x = np.stack(list_x, axis=0)
        y = np.asarray(list_y)

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)
