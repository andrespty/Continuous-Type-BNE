import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, alpha_0, r):
        super().__init__()
        self.alpha_0 = alpha_0
        self.r = r

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.alpha_0 / tf.pow(step+1, self.r)
    
    def get_config(self):
        return {
            'alpha_0': self.alpha_0,
            'r': self.r
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class CustomSchedule1(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, alpha_0, a, b, r):
        super().__init__()
        self.alpha_0 = alpha_0
        self.a = a
        self.b = b
        self.r = r

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.alpha_0 /(self.a + self.b * tf.pow(step, self.r))
    
    def get_config(self):
        return {
            'alpha_0': self.alpha_0,
            'a': self.a,
            'b': self.b,
            'r': self.r
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)