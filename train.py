import os
import tensorflow as tf
from tqdm import tqdm
from models.MLP_model import MLPMixerModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Normalization, RandomFlip, RandomRotation, RandomZoom, \
    Resizing
from loader import Loader


class MakePatches:
    def __init__(self, patch_size):
        super(MakePatches, self).__init__()
        self.patch_size = patch_size

    def __call__(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, self.patch_size, self.patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID")
        patches = tf.reshape(patches, [batch_size, -1, 3 * self.patch_size ** 2])  # (batch_size, patches, C)
        return patches


class Trainer:
    def __init__(self, train_data, val_data, C, DC, S, DS, classes, image_size,
                 learning_rate=0.001,
                 patch_size=32,
                 n_block_mlp_mixer=8,
                 batch_size=32,
                 epochs=32,
                 val_size=0.2):
        self.train_data = train_data
        self.val_data = val_data
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size

        assert (image_size * image_size) % (
                patch_size * patch_size) == 0, "Make sure the image size is dividable by patch size"

        self.augments = Sequential([
            Resizing(height=image_size, width=image_size),
            Normalization(),
            RandomFlip(),
            RandomRotation(factor=0.02),
            RandomZoom(height_factor=0.2, width_factor=0.2)
        ])
        self.patches = MakePatches(patch_size)

        self.model = MLPMixerModel(C, DC, S, DS, classes, image_size, n_block_mlp_mixer)
        self.optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        self.loss_object = SparseCategoricalCrossentropy()
        self.accuracy = SparseCategoricalAccuracy()
        self.loss = Mean()

        # Initialize check point
        self.saved_checkpoint = os.getcwd() + "/saved_checkpoint/"
        if not os.path.exists(self.saved_checkpoint):
            os.mkdir(self.saved_checkpoint)
        ckpt = tf.train.Checkpoint(transformer=self.model,
                                   optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.saved_checkpoint, max_to_keep=5)

    def train(self):
        # train_ds = image_dataset_from_directory(self.train_data,
        #                                         subset="training",
        #                                         seed=22,
        #                                         image_size=(self.image_size, self.image_size),
        #                                         batch_size=self.batch_size,
        #                                         validation_split=self.val_size,
        #                                         shuffle=True)
        #
        # val_ds = image_dataset_from_directory(self.val_data,
        #                                       subset="validation",
        #                                       seed=22,
        #                                       image_size=(self.image_size, self.image_size),
        #                                       batch_size=self.batch_size,
        #                                       validation_split=self.val_size,
        #                                       shuffle=True)
        # self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["acc"])
        # self.model.fit(train_ds, batch_size=self.batch_size, epochs=self.epochs, validation_data=val_ds)

        for epoch in range(epochs):
            X_train, Y_train = Loader(self.train_data, batch_size=self.batch_size, image_size=self.image_size).build()
            pbar = tqdm(enumerate(zip(X_train, Y_train)), total=len(X_train))
            for iter, (x, y) in pbar:
                with tf.GradientTape() as tape:
                    x = self.patches(x)
                    pred = self.model(x)
                    loss = self.loss_object(y, pred)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                self.accuracy(y, pred)
                self.loss(loss)

                pbar.set_description(
                    "Epoch {}  |  Loss: {:.4f}  |  Acc: {:.4f}".format(epoch, self.loss.result(),
                                                                       self.accuracy.result()))

            self.ckpt_manager.save()

    def predict(self, image_path):
        images = load_img(image_path)
        images = img_to_array(images)[tf.newaxis, ...]
        images = tf.image.resize(images, size=(self.image_size, image_size))
        self.ckpt_manager.restore_or_initialize()
        return self.model.predict(images)


if __name__ == '__main__':
    train_path = "dataset/train"
    val_path = "dataset/val"
    image_size = 224
    patch_size = 32
    batch_size = 20
    epochs = 2
    classes = 2
    n_blocks = 8
    C = 512
    DC = 2048
    S = (image_size * image_size) // (patch_size * patch_size)
    DS = 256
    trainer = Trainer(train_data=train_path,
                      val_data=val_path,
                      C=C, DC=DC, S=S, DS=DS,
                      classes=classes, image_size=image_size,
                      patch_size=patch_size, batch_size=batch_size,
                      epochs=epochs, n_block_mlp_mixer=n_blocks)

    test_image = "dataset/train/faces/00000.jpg"
    trainer.train()
    print(trainer.predict(test_image))
