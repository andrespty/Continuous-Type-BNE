from abc import ABC, abstractmethod


class Game(ABC):
    """
    Abstract base class for all games.

    A Game defines the structure of the equilibrium problem:
    how many players, what the type and action spaces look like,
    how many dual variables are needed, and which loss encodes
    the equilibrium conditions.

    The Solver uses these properties so that game-specific
    details never leak into training code.
    """

    @abstractmethod
    def nn_model(self):
        """
        Return a tf.keras.Model that takes type samples as input
        and outputs primal and dual actions.
        Called once by the Solver at compile time.
        """
        ...

    @abstractmethod
    def loss_fn(self):
        """
        Return the equilibrium loss function for this game.
        Called once by the Solver at compile time.
        """
        ...

    @abstractmethod
    def optimizer(self, lr=1e-5, isAdam=True):
        """
        Return the optimizer to use during training.
        Called once by the Solver at compile time.
        """
        ...

    def custom_objects(self):
        """
        Return a dictionary of custom objects (losses, optimizers, etc.)
        needed to load a saved model for this game.
        """
        return {
            'loss_fn': self.loss_fn(),
            'optimizer': self.optimizer(),
        }