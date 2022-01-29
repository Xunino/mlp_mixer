import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class CustomSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_step=4000):
        super(CustomSchedule, self).__init__()
        self.warmup_step = warmup_step
        self.d_model = tf.cast(d_model, dtype=tf.float32)

    def __call__(self, step):
        result = self.d_model ** -0.5 * tf.minimum(step ** -0.5, step * self.warmup_step ** -1.5)
        return result
