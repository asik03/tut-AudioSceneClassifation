import os


if __name__ == '__main__':
    # TODO: Read from ./etc/raw_data/meta.txt the audios' metadata and rename the files
    raw_data_dir = './etc/raw_data/'
    meta_file = os.path.join(raw_data_dir, 'meta.txt')

    for line in open(meta_file):
        # TODO: Extract info from meta.txt here
        pass

    file_names = os.listdir(raw_data_dir)
    file_names = list(filter(lambda x: x.endswith('.wav'), file_names))

    for file_name in file_names:
        file_path = os.path.join(raw_data_dir, file_name)
        # TODO: Rename the file here with the info from meta.txt
