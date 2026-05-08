import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf

class EquilibriumLoss(tf.keras.losses.Loss):
    def __init__(self, name='equilibrium_loss', **kwargs):
        self.n_players = kwargs.pop('n_players', 2)
        self.isSymmetric = kwargs.pop('isSymmetric', True)
        kwargs.pop('reduction', None)
        super().__init__(reduction=None, name=name, **kwargs)
    
    def complementarity_func(self, a, b):
        return tf.sqrt(tf.square(a) + tf.square(b)) - (a + b)

    def primal_energy(self, types, actions):
        pass

    def dual_energy(self, types, actions):
        pass

    def call(self, types, actions):
        eng = self.primal_energy(types, actions)
        dual_eng = self.dual_energy(types, actions)

        if self.isSymmetric:
            return self.n_players * (eng + dual_eng)
        else:
            return eng + dual_eng
    
    def get_config(self):
        config = super().get_config()
        # add any custom params here if you have them
        return config