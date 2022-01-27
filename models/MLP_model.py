import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, GlobalAvgPool1D, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization, RandomFlip, RandomRotation, RandomZoom, \
    Resizing


class MLPMixer(Layer):
    def __init__(self, C, DC, S, DS):
        """
        :param C: Hidden_dims
        :param DC: MLP_dims (2048)
        :param S: Patches (HW / P ** 2)
        :param DS: MPL_dims (256)
        """
        super(MLPMixer, self).__init__()
        w_init = tf.random_normal_initializer()

        self.layerNorm_channel = LayerNormalization()
        self.layerNorm_token = LayerNormalization()

        # Channel mixing: (C)
        # Input: (batch_size, patches, C)
        # Weight_1: (C, DC) - C: Hidden_dims (512) -- DC: MLP_dims (2048) -> (batch_size, patches, DC)
        self.W_1 = tf.Variable(initial_value=w_init(shape=(C, DC), dtype=tf.float32), trainable=True, name="w1")
        # Weight_2: (DC, C) - C: Hidden_dims (512) -- DC: MLP_dims (2048) -> (batch_size, patches, C)
        self.W_2 = tf.Variable(initial_value=w_init(shape=(DC, C), dtype=tf.float32), trainable=True, name="w2")

        # Token mixing: (S)
        # Input: (batch_size, S, C) -> transpose -> (batch_size, C, S) -- S is patches = (HW/P**2)
        # Weight_3: (S, DS) - S: (HW/P**2) -- DS: MLP_dims (256) -> (batch_size, C, DS)
        self.W_3 = tf.Variable(initial_value=w_init(shape=(S, DS), dtype=tf.float32), trainable=True, name="w3")
        # Weight_4: (DS, S) - S: (HW/P**2) -- DS: MLP_dims (256) -> (batch_size, C, S) -> transpose-> (batch_size, S, C)
        self.W_4 = tf.Variable(initial_value=w_init(shape=(DS, S), dtype=tf.float32), trainable=True, name="w4")

    def __call__(self, x, *args, **kwargs):
        # Token mixing
        x = self.token_mixing(x)
        # Channel mixing
        x = self.channel_mixing(x)
        return x

    def channel_mixing(self, x):
        norm_out = self.layerNorm_channel(x)
        out = tf.matmul(norm_out, self.W_1)
        out = tf.nn.gelu(out)
        out = tf.matmul(out, self.W_2) + x
        return out

    def token_mixing(self, x):
        norm_out = self.layerNorm_token(x)
        out = tf.transpose(norm_out, perm=(0, 2, 1))  # (BS, C, S)
        out = tf.matmul(out, self.W_3)
        out = tf.nn.gelu(out)
        out = tf.matmul(out, self.W_4)
        out = tf.transpose(out, perm=(0, 2, 1)) + x  # (BS, S, C)
        return out


class MLPMixerModel(Model):
    def __init__(self, C, DC, S, DS, num_classes, image_size=256, patch_size=32, n_block_mlp_mixer=8, name=None):
        """
        :param C: Hidden dims
        :param DC: MLP_dims (2048)
        :param S: Patches (token) (HW/P**2)
        :param DS: MLP_dims (256)
        :param num_classes:
        :param patch_size: (32, 32)
        :param n_block_mlp_mixer: (8)
        """
        super(MLPMixerModel, self).__init__(name=name)
        self.augments = Sequential([
            Resizing(height=image_size, width=image_size),
            Normalization(),
            RandomFlip(),
            RandomRotation(factor=0.02),
            RandomZoom(height_factor=0.2, width_factor=0.2)
        ])

        self.patch_size = patch_size

        self.projection = Dense(C)
        self.blocks = [MLPMixer(C, DC, S, DS) for _ in range(n_block_mlp_mixer)]

        self.classification = Sequential([
            GlobalAvgPool1D(),
            Dropout(0.2),
            Dense(num_classes, activation="softmax")
        ])

    def __call__(self, x, *args, **kwargs):
        x = self.augments(x)
        x = self.patches(x, self.patch_size)
        x = self.projection(x)
        for block in self.blocks:
            x = block(x)

        x = self.classification(x)
        return x

    def patches(self, images, patch_size):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images,
                                           sizes=[1, patch_size, patch_size, 1],
                                           strides=[1, patch_size, patch_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID")
        patches = tf.reshape(patches, [batch_size, -1, 3 * patch_size ** 2])  # (batch_size, patches, C)
        return patches


if __name__ == '__main__':
    image_size = 224
    patch_size = 32
    C = 512
    DC = 2048
    DS = 256
    S = (image_size * image_size) // (patch_size * patch_size)
    num_classes = 1000
    n_blocks = 8
    images = tf.random.uniform(shape=(1, 256, 256, 3), maxval=1.)
    mlp = MLPMixerModel(C, DC, S, DS, num_classes, image_size, patch_size, n_blocks)
    x = mlp(images)
    print(x.shape)
