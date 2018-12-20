import os
import pickle
from src.cnn.scene_network import SceneNetwork

if __name__ == '__main__':
    real_dir = 'D:\\UNIVERSIDAD\\MASTER\\(2) SEGUNDO\\Advanced Signal Processing Laboratory\\AUDIO\\etc\\real'

    pickle_dir = './etc/features'

    labels = ['bus', 'cafe_restaurant', 'beach', 'city_center', 'forest_path', 'car', 'grocery_store', 'home',
              'library', 'metro_station', 'office', 'park', 'residential_area', 'train', 'tram']

    batch_size = 32

    sampling_rate = 44100.0
    hop_size = 512.0
    time_duration = 10.0
    t_dim = (int(time_duration * sampling_rate / hop_size) + 1) * 2.0

    n_features = 20
    method = 'both'

    real_pickle_name = 'real_' + str(n_features) + '_' + method + '.pkl'

    input_shape = (n_features, t_dim, 1)

    scene_network = SceneNetwork()
    scene_network.build_model(input_shape)

    with open(os.path.join(pickle_dir, real_pickle_name), 'rb') as f:
        x_eval, y_eval = pickle.load(f)

    scene_network.evaluate_model(x_eval, y_eval)
