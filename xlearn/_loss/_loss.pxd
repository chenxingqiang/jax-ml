import jax
import jax.numpy as jnp

class DoublePair:
    def __init__(self, val1, val2):
        self.val1 = val1
        self.val2 = val2

def cy_loss(y_true, raw_prediction):
    ...

def cy_gradient(y_true, raw_prediction):
    ...

def cy_grad_hess(y_true, raw_prediction):
    ...

class CyLossFunction:
    def __init__(self):
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)


class CyHalfSquaredError:
    def __init__(self):
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)

class CyAbsoluteError:
    def __init__(self):
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)

class CyPinballLoss:
    def __init__(self, quantile):
        self.quantile = quantile
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)

class CyHuberLoss:
    def __init__(self, delta):
        self.delta = delta
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)

class CyHalfPoissonLoss:
    def __init__(self):
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)

class CyHalfGammaLoss:
    def __init__(self):
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)

class CyHalfTweedieLoss:
    def __init__(self, power):
        self.power = power
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)

class CyHalfTweedieLossIdentity:
    def __init__(self, power):
        self.power = power
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)

class CyHalfBinomialLoss:
    def __init__(self):
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)

class CyExponentialLoss:
    def __init__(self):
        self.cy_loss = jax.jit(cy_loss)
        self.cy_gradient = jax.jit(cy_gradient)
        self.cy_grad_hess = jax.jit(cy_grad_hess)
