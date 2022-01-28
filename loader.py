import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


class Loader:
    def __init__(self, image_path, batch_size=32, image_size=256):
        self.image_path = image_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.autotune = tf.data.experimental.AUTOTUNE
        self.list_image_paths = []
        self.list_image_labels = []
        self.get_list_images()

    def get_list_images(self):
        for dir_names in os.listdir(self.image_path):
            for file in os.listdir(os.path.join(self.image_path, dir_names)):
                self.list_image_paths.append(os.path.join(self.image_path, dir_names, file))
                self.list_image_labels.append(dir_names)

        lb = LabelEncoder()
        lb.fit(np.unique(self.list_image_labels))
        self.list_image_labels = lb.transform(self.list_image_labels)
        self.list_image_paths, self.list_image_labels = shuffle(self.list_image_paths, self.list_image_labels,
                                                                random_state=42)

    def processing_image(self, file_image):
        img = tf.io.read_file(file_image)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.image_size, self.image_size])
        return img

    def config_for_image_performance(self, ds):
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.autotune)
        return ds

    def build(self):
        train_ds = tf.data.Dataset.list_files(self.list_image_paths)
        train_ds = train_ds.map(self.processing_image, num_parallel_calls=self.autotune)

        # Performance
        return self.config_for_image_performance(train_ds), self.config_for_image_performance(
            tf.data.Dataset.from_tensor_slices(self.list_image_labels))


if __name__ == '__main__':
    path = "dataset/train/"
    loader = Loader(path)
    _, labels = loader.build()
    print(len(labels))
    print(next(iter(labels)))
