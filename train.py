import os
import tensorflow as tf
from tqdm import tqdm
from loader import Loader
from metrics import CustomSchedule
from argparse import ArgumentParser
from models.MLP_model import MLPMixerModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers.experimental.preprocessing import Normalization, RandomFlip, RandomRotation, RandomZoom


class Trainer:
    def __init__(self, C, DC, DS,
                 train_path,
                 val_path,
                 classes,
                 image_size=224,
                 learning_rate=0.001,
                 patch_size=32,
                 n_block_mlp_mixer=8,
                 batch_size=32,
                 epochs=32,
                 val_size=0.2,
                 augments=None,
                 retrain=False):
        self.train_path = train_path
        self.val_path = val_path
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size

        assert (image_size * image_size) % (
                patch_size * patch_size) == 0, "Make sure the image size is dividable by patch size"
        S = (args.image_size * args.image_size) // (args.patch_size * args.patch_size)

        if augments is None:
            self.augments = Sequential([Normalization(),
                                        RandomFlip(),
                                        RandomRotation(factor=0.02),
                                        RandomZoom(height_factor=0.2, width_factor=0.2)])
        else:
            self.augments = augments

        if retrain:
            lr = learning_rate
        else:
            lr = CustomSchedule(C)

        self.model = MLPMixerModel(C, DC, S, DS, classes, patch_size, n_block_mlp_mixer)
        self.optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
        self.loss_object = SparseCategoricalCrossentropy(from_logits=True)
        self.train_acc_metric = SparseCategoricalAccuracy(name="train")
        self.val_acc_metric = SparseCategoricalAccuracy(name="val")

        # Initialize check point
        self.saved_checkpoint = os.getcwd() + "/saved_checkpoint/"
        if not os.path.exists(self.saved_checkpoint):
            os.mkdir(self.saved_checkpoint)
        ckpt = tf.train.Checkpoint(transformer=self.model,
                                   optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.saved_checkpoint, max_to_keep=5)

        if retrain:
            print("[INFO] Retrain...")
            print("[INFO] Loaded model.")
            self.ckpt_manager.restore_or_initialize()
            print("[INFO] Start training...")

    def train(self):
        for epoch in range(self.epochs):
            x_train, y_train = Loader(self.train_path, batch_size=self.batch_size, image_size=self.image_size).build()
            pbar = tqdm(enumerate(zip(x_train, y_train)), total=len(x_train))
            for iter, (x, y) in pbar:
                loss = self.train_step(x, y)

                if self.val_path is not None:
                    x_val, y_val = Loader(self.val_path, batch_size=self.batch_size, image_size=self.image_size).build()
                    for _x, _y in zip(x_val, y_val):
                        self.val_step(x, y)
                    description = "Epoch {}  |  Loss: {:.4f}  |  Acc: {:.4f}  |  Val_acc: {:.4f}  ".format(epoch + 1,
                                                                                                           loss,
                                                                                                           self.train_acc_metric.result(),
                                                                                                           self.val_acc_metric.result())
                else:
                    description = "Epoch {}  |  Loss: {:.4f}  |  Acc: {:.4f}  ".format(epoch + 1, loss,
                                                                                       self.train_acc_metric.result())

                pbar.set_description(description)

            self.train_acc_metric.reset_state()
            self.val_acc_metric.reset_state()
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

    @tf.function
    def val_step(self, x, y):
        y_pred = self.model(x, training=False)
        self.val_acc_metric.update_state(y, y_pred)

    def predict(self, image_path):
        images = load_img(image_path)
        images = img_to_array(images)[tf.newaxis, ...]
        images = tf.image.resize(images, size=(self.image_size, self.image_size))
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
    """
    python train.py --train-path=dataset/train --val-path=dataset/val --epochs=20
    """
    parser = ArgumentParser()

    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--train-path", required=True, type=str)
    parser.add_argument("--val-path", default=None, type=str)
    parser.add_argument("--classes", default=2, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--n_blocks", default=8, type=int)
    parser.add_argument("--C", default=512, type=int)
    parser.add_argument("--DC", default=1024, type=int)
    parser.add_argument("--DS", default=256, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--patch-size", default=32, type=int)
    parser.add_argument("--augments", default=False, type=bool)
    parser.add_argument("--retrain", default=False, type=bool)

    args = parser.parse_args()

    # FIXME
    # Project Description
    print('---------------------Welcome to Hợp tác xã Kiên trì-------------------')
    print('Github: https://github.com/Xunino')
    print('Email : ndlinh.ai@gmail.com')
    print('------------------------------------------------------------------------')
    print(f'MLP Mixer model with hyper-params:')
    print('------------------------------------')
    for k, v in vars(args).items():
        print(f"|  +) {k} = {v}")
    print('====================================')

    trainer = Trainer(train_path=args.train_path,
                      val_path=args.val_path,
                      C=args.C, DC=args.DC, DS=args.DS,
                      classes=args.classes, image_size=args.image_size,
                      patch_size=args.patch_size, batch_size=args.batch_size,
                      epochs=args.epochs, n_block_mlp_mixer=args.n_blocks, augments=args.augments, retrain=args.retrain)
    trainer.train()
