import os
import pickle
from src.utilities.features_loader import FeaturesLoader

if __name__ == '__main__':
    train_dir = 'D:\\UNIVERSIDAD\\MASTER\\(2) SEGUNDO\\Advanced Signal Processing Laboratory\\AUDIO\\etc\\train'
    validation_dir = 'D:\\UNIVERSIDAD\\MASTER\\(2) SEGUNDO\\Advanced Signal Processing Laboratory\\AUDIO\\etc\\validation'
    evaluation_dir = 'D:\\UNIVERSIDAD\\MASTER\\(2) SEGUNDO\\Advanced Signal Processing Laboratory\\AUDIO\\etc\\evaluation'
    real_dir = 'D:\\UNIVERSIDAD\\MASTER\\(2) SEGUNDO\\Advanced Signal Processing Laboratory\\AUDIO\\etc\\real'

    pickle_dir = './etc/features'

    labels = ['bus', 'cafe_restaurant', 'beach', 'city_center', 'forest_path', 'car', 'grocery_store', 'home',
              'library', 'metro_station', 'office', 'park', 'residential_area', 'train', 'tram']

    n_features = 20
    method = 'both'

    train_pickle_name = 'train_' + str(n_features) + '_' + method + '.pkl'
    validation_pickle_name = 'validation_' + str(n_features) + '_' + method + '.pkl'
    evaluation_pickle_name = 'evaluation_' + str(n_features) + '_' + method + '.pkl'
    real_pickle_name = 'real_' + str(n_features) + '_' + method + '.pkl'

    train_loader = FeaturesLoader(train_dir, labels, n_features=n_features, method=method)
    validation_loader = FeaturesLoader(validation_dir, labels, n_features=n_features, method=method)
    evaluation_loader = FeaturesLoader(evaluation_dir, labels, n_features=n_features, method=method)
    real_loader = FeaturesLoader(real_dir, labels, n_features=n_features, method=method)

    x_train, y_train = train_loader.get_data()
    with open(os.path.join(pickle_dir, train_pickle_name), 'wb') as f:
        pickle.dump((x_train, y_train), f)

    x_val, y_val = validation_loader.get_data()
    with open(os.path.join(pickle_dir, validation_pickle_name), 'wb') as f:
        pickle.dump((x_val, y_val), f)

    x_eval, y_eval = evaluation_loader.get_data()
    with open(os.path.join(pickle_dir, evaluation_pickle_name), 'wb') as f:
        pickle.dump((x_eval, y_eval), f)

    x_real, y_real = real_loader.get_data()
    with open(os.path.join(pickle_dir, real_pickle_name), 'wb') as f:
        pickle.dump((x_real, y_real), f)
