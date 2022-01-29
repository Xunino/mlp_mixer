import os
import tensorflow as tf
from tqdm import tqdm
from models.MLP_model import MLPMixerModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from loader import Loader
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Normalization, RandomFlip, RandomRotation, RandomZoom


class Trainer:
    def __init__(self, train_data, C, DC, S, DS, classes, image_size,
                 learning_rate=0.001,
                 patch_size=32,
                 n_block_mlp_mixer=8,
                 batch_size=32,
                 epochs=32,
                 val_size=0.2,
                 augments=None):
        self.train_data = train_data
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size

        assert (image_size * image_size) % (
                patch_size * patch_size) == 0, "Make sure the image size is dividable by patch size"

        if augments is None:
            self.augments = Sequential([Normalization(),
                                        RandomFlip(),
                                        RandomRotation(factor=0.02),
                                        RandomZoom(height_factor=0.2, width_factor=0.2)])
        else:
            self.augments = augments

        self.model = MLPMixerModel(C, DC, S, DS, classes, patch_size, n_block_mlp_mixer)
        self.optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        self.loss_object = SparseCategoricalCrossentropy(from_logits=True)

        self.train_acc_metric = SparseCategoricalAccuracy()

        # Initialize check point
        self.saved_checkpoint = os.getcwd() + "/saved_checkpoint/"
        if not os.path.exists(self.saved_checkpoint):
            os.mkdir(self.saved_checkpoint)
        ckpt = tf.train.Checkpoint(transformer=self.model,
                                   optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.saved_checkpoint, max_to_keep=5)

    def train(self):
        for epoch in range(epochs):
            x_train, y_train = Loader(self.train_data, batch_size=self.batch_size, image_size=self.image_size).build()
            pbar = tqdm(enumerate(zip(x_train, y_train)), total=len(x_train))
            for iter, (x, y) in pbar:
                loss = self.train_step(x, y)

                pbar.set_description(
                    "Epoch {}  |  Loss: {:.4f}  |  Acc: {:.4f}  ".format(epoch, loss, self.train_acc_metric.result()))

            self.train_acc_metric.reset_state()
        self.ckpt_manager.save()

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            if self.augments:
                x = self.augments(x)
            pred = self.model(x, training=True)
            loss = self.loss_object(y, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_acc_metric.update_state(y, pred)
        return loss

    def predict(self, image_path):
        images = load_img(image_path)
        images = img_to_array(images)[tf.newaxis, ...]
        images = tf.image.resize(images, size=(self.image_size, image_size))
        self.ckpt_manager.restore_or_initialize()
        return self.model.predict(images)


def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 3)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


if __name__ == '__main__':
    train_path = "dataset/train"
    image_size = 224
    patch_size = 32
    batch_size = 4
    epochs = 20
    classes = 2
    n_blocks = 8
    C = 512
    DC = 2048
    S = (image_size * image_size) // (patch_size * patch_size)
    DS = 256
    augments = False
    trainer = Trainer(train_data=train_path,
                      C=C, DC=DC, S=S, DS=DS,
                      classes=classes, image_size=image_size,
                      patch_size=patch_size, batch_size=batch_size,
                      epochs=epochs, n_block_mlp_mixer=n_blocks, augments=augments)

    test_image = "dataset/train/faces/00000.jpg"
    trainer.train()
    print(trainer.predict(test_image))
