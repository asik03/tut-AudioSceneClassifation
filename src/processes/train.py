import os
from src.cnn.scene_network import SceneNetwork
from src.utilities.features_generator import FeaturesGenerator

if __name__ == '__main__':
    train_dir = './etc/train'
    validation_dir = './etc/validation'
    evaluation_dir = './etc/evaluation'

    labels = ['bus', 'cafe_restaurant', 'beach', 'city_center', 'forest_path', 'car', 'grocery_store', 'home',
              'library', 'metro_station', 'office', 'park', 'residential_area', 'train', 'tram']

    batch_size = 32

    sampling_rate = 44100.0
    hop_size = 512.0
    time_duration = 10.0
    t_dim = int(time_duration * sampling_rate / hop_size)

    n_features = 20

    input_shape = (batch_size, n_features, t_dim)

    train_generator = FeaturesGenerator(train_dir, labels, n_features=n_features, batch_size=batch_size)
    validation_generator = FeaturesGenerator(validation_dir, labels, n_features=n_features, batch_size=batch_size)
    evaluation_generator = FeaturesGenerator(evaluation_dir, labels, n_features=n_features, batch_size=batch_size)

    scene_network = SceneNetwork()
    scene_network.build_model(input_shape)

    scene_network.train_model(train_generator=train_generator,
                              validation_generator=validation_generator)

    scene_network.save_model('./etc/models/model.h5')
    scene_network.evaluate_model(evaluate_generator=evaluation_generator)
