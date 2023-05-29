import jax.numpy as np
from jax import vmap
from typing import Tuple

class LossFunction:
    def loss(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def gradient(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad_hess(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class HalfSquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def gradient(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def grad_hess(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


# Repeat above code for AbsoluteError, PinballLoss, HuberLoss, HalfPoissonLoss, HalfGammaLoss,
# HalfTweedieLoss, HalfTweedieLossIdentity, HalfBinomialLoss, ExponentialLoss
class AbsoluteError(LossFunction):
    def loss(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return np.abs(y_true - raw_prediction)

    def gradient(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return np.sign(raw_prediction - y_true)

    def grad_hess(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.gradient(y_true, raw_prediction), np.zeros_like(raw_prediction)


class PinballLoss(LossFunction):
    def __init__(self, quantile):
        self.quantile = quantile

    def loss(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        diff = y_true - raw_prediction
        return np.where(diff >= 0, self.quantile * diff, (self.quantile - 1) * diff)

    # Gradient and Hessian need to be defined according to the formula of Pinball Loss


class HuberLoss(LossFunction):
    def __init__(self, delta):
        self.delta = delta

    def loss(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return np.where(np.abs(y_true - raw_prediction) <= self.delta, 
                        0.5 * np.square(y_true - raw_prediction), 
                        self.delta * (np.abs(y_true - raw_prediction) - 0.5 * self.delta))

    # Gradient and Hessian need to be defined according to the formula of Huber Loss


# Similar pattern should be followed for other classes:
# HalfPoissonLoss, HalfGammaLoss, HalfTweedieLoss, HalfTweedieLossIdentity, HalfBinomialLoss, ExponentialLoss
class HalfPoissonLoss(LossFunction):
    def loss(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return np.exp(raw_prediction) - y_true * raw_prediction

    def gradient(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return np.exp(raw_prediction) - y_true

    def grad_hess(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.gradient(y_true, raw_prediction), np.exp(raw_prediction)


class HalfGammaLoss(LossFunction):
    # Assuming the canonical link function is used (log link)
    def loss(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return np.exp(-raw_prediction) * y_true + raw_prediction

    def gradient(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return -np.exp(-raw_prediction) * y_true + 1

    def grad_hess(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.gradient(y_true, raw_prediction), np.exp(-raw_prediction) * y_true


class HalfTweedieLoss(LossFunction):
    # Assuming the power is an attribute of the class
    def __init__(self, power):
        self.power = power

    def loss(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        raise NotImplementedError  # Depends on the specific power parameter

    # The gradient and Hessian depend on the specific power parameter


class HalfTweedieLossIdentity(HalfTweedieLoss):
    # Inherits from HalfTweedieLoss, so no need to redefine __init__ or loss


class HalfBinomialLoss(LossFunction):
    def loss(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(raw_prediction)) - y_true * raw_prediction

    def gradient(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return np.exp(raw_prediction) / (1 + np.exp(raw_prediction)) - y_true

    def grad_hess(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        exp_raw = np.exp(raw_prediction)
        return self.gradient(y_true, raw_prediction), exp_raw / ((1 + exp_raw) ** 2)


class ExponentialLoss(LossFunction):
    def loss(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return np.exp(raw_prediction) - y_true * raw_prediction

    def gradient(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> np.ndarray:
        return np.exp(raw_prediction) - y_true

    def grad_hess(self, y_true: np.ndarray, raw_prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.gradient(y_true, raw_prediction), np.exp(raw_prediction)
