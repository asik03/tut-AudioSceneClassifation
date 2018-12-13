import os

"""
Raw data must be saved as:
│       ├── etc
│   ├── raw_data
├── audio
│audio_samples.wav
│   ├── meta.txt
"""

if __name__ == '__main__':
    raw_data_dir = '.\\etc\\raw_data\\'
    meta_file = os.path.join(raw_data_dir, 'meta.txt')

    audio_names = []
    audio_labels = []

    for count, line in enumerate(open(meta_file)):
        line = line.replace(' ', '\t')
        a = line.split("\t")
        b = a[0].split("/")
        audio_names.append(b[1])
        audio_labels.append(a[1])

    audio_labels = [label.replace('/', '_') for label in audio_labels]

    raw_data_audio_dir = os.path.join(raw_data_dir, "audio\\")
    file_names = os.listdir(raw_data_audio_dir)
    file_names = list(filter(lambda x: x.endswith('.wav'), file_names))

    for count, file_name in enumerate(file_names):
        index = audio_names.index(file_name)
        name = file_name.split(".")
        new_name = [name[0], "-", audio_labels[index], ".wav"]
        new_name = ''.join(new_name)
        new_file_path = os.path.join(raw_data_audio_dir, new_name)
        old_file_path = os.path.join(raw_data_audio_dir, file_name)
        os.rename(old_file_path, new_file_path)
