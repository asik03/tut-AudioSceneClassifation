import os
import pickle
from src.cnn.scene_network import SceneNetwork

if __name__ == '__main__':
    train_dir = 'D:\\UNIVERSIDAD\\MASTER\\(2) SEGUNDO\\Advanced Signal Processing Laboratory\\AUDIO\\etc\\train'
    validation_dir = 'D:\\UNIVERSIDAD\\MASTER\\(2) SEGUNDO\\Advanced Signal Processing Laboratory\\AUDIO\\etc\\validation'
    evaluation_dir = 'D:\\UNIVERSIDAD\\MASTER\\(2) SEGUNDO\\Advanced Signal Processing Laboratory\\AUDIO\\etc\\evaluation'

    pickle_dir = './etc/features'

    labels = ['bus', 'cafe_restaurant', 'beach', 'city_center', 'forest_path', 'car', 'grocery_store', 'home',
              'library', 'metro_station', 'office', 'park', 'residential_area', 'train', 'tram']

    batch_size = 32

    sampling_rate = 44100.0
    hop_size = 512.0
    time_duration = 10.0
    t_dim = int(time_duration * sampling_rate / hop_size) + 1

    n_features = 20
    method = 'mfcc'

    train_pickle_name = 'train_' + str(n_features) + '_' + method + '.pkl'
    validation_pickle_name = 'validation_' + str(n_features) + '_' + method + '.pkl'
    evaluation_pickle_name = 'evaluation_' + str(n_features) + '_' + method + '.pkl'

    input_shape = (n_features, t_dim, 1)

    scene_network = SceneNetwork()
    scene_network.build_model(input_shape)

    with open(os.path.join(pickle_dir, train_pickle_name)) as f:
        x_train, y_train = pickle.load(f)

    with open(os.path.join(pickle_dir, validation_pickle_name)) as f:
        x_val, y_val = pickle.load(f)

    scene_network.train_model(x_train, y_train, x_val, y_val, batch_size=batch_size, epochs=50)
    scene_network.save_model('./etc/models/model.h5')

    # Clear dataset from memory
    del x_train
    del y_train
    del x_val
    del y_val

    with open(os.path.join(pickle_dir, evaluation_pickle_name)) as f:
        x_eval, y_eval = pickle.load(f)

    scene_network.evaluate_model(x_eval, y_eval)
