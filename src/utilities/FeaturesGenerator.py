import os
import numpy as np
import keras


class FeaturesGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, data_dir, labels, batch_size=32, n_classes=15, shuffle=True, method='mfcc'):
        """Initialization"""
        self.batch_size = batch_size
        self.labels = labels
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.file_names = self.__get_file_names()
        self.indexes = self.__get_indexes()
        self.method = method

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

        # Find list of IDs
        # list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        # x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = self.__get_indexes()

    def __data_generation(self, list_ids_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        # Initialization

        # return x, keras.utils.to_categorical(y, num_classes=self.n_classes)
