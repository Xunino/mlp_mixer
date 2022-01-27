import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, LayerNormalization


class SplitPatch(Layer):
    def __init__(self, P):
        super(SplitPatch, self).__init__()
        self.P = P

    def __call__(self, images, *args, **kwargs):
        batch_size = images.shape[0]
        patches = tf.image.extract_patches(images,
                                           sizes=[1, self.P, self.P, 1],
                                           strides=[1, self.P, self.P, 1],
                                           rates=[1, 1, 1, 1],
                                           padding="VALID")
        patches = tf.reshape(patches, shape=(batch_size, -1, 3 * self.P ** 2))  # (batch_size, patches, C)
        return patches


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
        self.W_1 = tf.Variable(initial_value=w_init(shape=(C, DC), dtype=tf.float32), trainable=True)
        # Weight_2: (DC, C) - C: Hidden_dims (512) -- DC: MLP_dims (2048) -> (batch_size, patches, C)
        self.W_2 = tf.Variable(initial_value=w_init(shape=(DC, C), dtype=tf.float32), trainable=True)

        # Token mixing: (S)
        # Input: (batch_size, S, C) -> transpose -> (batch_size, C, S) -- S is patches = (HW/P**2)
        # Weight_3: (S, DS) - S: (HW/P**2) -- DS: MLP_dims (256) -> (batch_size, C, DS)
        self.W_3 = tf.Variable(initial_value=w_init(shape=(S, DS), dtype=tf.float32), trainable=True)
        # Weight_4: (DS, S) - S: (HW/P**2) -- DS: MLP_dims (256) -> (batch_size, C, S) -> transpose-> (batch_size, S, C)
        self.W_4 = tf.Variable(initial_value=w_init(shape=(DS, S), dtype=tf.float32), trainable=True)

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
    def __init__(self, ):
        super(MLPMixerModel, self).__init__()
        self.classification = None

    def __call__(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    C = 512
    DC = 2048
    S = (256 * 256) // (32 * 32)
    DS = 256
    images = tf.random.uniform(shape=(1, 256, 256, 3), maxval=1.)
    image_patches = SplitPatch(32)
    mlp = MLPMixer(C, DC, S, DS)
    x = image_patches(images)
    x = Dense(C)(x)
    x = mlp(x)
    print(x.shape)
