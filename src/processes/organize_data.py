import os
import random
import shutil

if __name__ == '__main__':
    raw_data_dir = './etc/raw_data'
    train_data_dir = './etc/train'
    validation_data_dir = './etc/validation'
    evaluation_data_dir = './etc/evaluation'

    percentage_eval = 0.2
    percentage_train = 0.7
    percentage_val = 0.1

    file_names = os.listdir(raw_data_dir)
    labels = ['bus', 'cafe/restaurant', 'beach', 'city_center', 'forest_path', 'car', 'grocery_store', 'home',
              'library', 'metro_station', 'office', 'park', 'residential_area', 'train', 'tram']

    file_names_classes = {}
    for label in labels:
        file_names_classes[label] = list(filter(lambda x: label in x, file_names))

        n_examples = len(file_names_classes[label])
        n_examples_eval = int(percentage_eval * n_examples)
        n_examples_train = int(percentage_train * n_examples)
        n_examples_val = n_examples - n_examples_eval - n_examples_train

        indexes = range(n_examples)
        random.shuffle(indexes)

        indexes_eval = indexes[0:n_examples_eval]
        indexes_train = indexes[n_examples_eval:n_examples_train]
        indexes_val = indexes[n_examples_eval + n_examples_train:]

        for indexes, path in zip((indexes_eval, indexes_train, indexes_val),
                                 (evaluation_data_dir, train_data_dir, validation_data_dir)):
            for index in indexes:
                shutil.copyfile(os.path.join(raw_data_dir, file_names_classes[label][index]), path)
