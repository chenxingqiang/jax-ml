import jax
import jax.numpy as jnp


class DoublePair:
    def __init__(self, val1, val2):
        self.val1 = val1
        self.val2 = val2

@jax.jit
def log1pexp(x):
    if x <= -37:
        return jnp.exp(x)
    elif x <= -2:
        return jnp.log1p(jnp.exp(x))
    elif x <= 18:
        return jnp.log(1. + jnp.exp(x))
    elif x <= 33.3:
        return x + jnp.exp(-x)
    else:
        return x

@jax.jit
def sum_exp_minus_max(i, raw_prediction):
    n_classes = raw_prediction.shape[1]
    max_value = raw_prediction[i, 0]
    sum_exps = 0.0
    for k in range(1, n_classes):
        if max_value < raw_prediction[i, k]:
            max_value = raw_prediction[i, k]
    p = jnp.zeros(n_classes + 2)
    for k in range(n_classes):
        p[k] = jnp.exp(raw_prediction[i, k] - max_value)
        sum_exps += p[k]
    p[n_classes] = max_value
    p[n_classes + 1] = sum_exps
    return p

@jax.jit
def closs_half_squared_error(y_true, raw_prediction):
    return 0.5 * (raw_prediction - y_true) * (raw_prediction - y_true)

@jax.jit
def cgradient_half_squared_error(y_true, raw_prediction):
    return raw_prediction - y_true

@jax.jit
def cgrad_hess_half_squared_error(y_true, raw_prediction):
    gh = jnp.zeros(2)
    gh[0] = raw_prediction - y_true
    gh[1] = 1.0
    return gh

@jax.jit
def closs_absolute_error(y_true, raw_prediction):
    return jnp.abs(raw_prediction - y_true)

@jax.jit
def cgradient_absolute_error(y_true, raw_prediction):
    return 1.0 if raw_prediction > y_true else -1.0

@jax.jit
def cgrad_hess_absolute_error(y_true, raw_prediction):
    gh = jnp.zeros(2)
    gh[0] = 1.0 if raw_prediction > y_true else -1.0
    gh[1] = 1.0
    return gh

@jax.jit
def closs_pinball_loss(y_true, raw_prediction, quantile):
    return (quantile * (y_true - raw_prediction) if y_true >= raw_prediction
            else (1. - quantile) * (raw_prediction - y_true))

@jax.jit
def cgradient_pinball_loss(y_true, raw_prediction, quantile):
    return -quantile if y_true >= raw_prediction else 1. - quantile

@jax.jit
def cgrad_hess_pinball_loss(y_true, raw_prediction, quantile):
    gh = jnp.zeros(2)
    gh[0] = -quantile if y_true >= raw_prediction else 1. - quantile
    gh[1] = 1.0
    return gh

@jax.jit
def closs_huber_loss(y_true, raw_prediction, delta):
    abserr = jnp.abs(y_true - raw_prediction)
    if abserr <= delta:
        return 0.5 * abserr**2
    else:
        return delta * (abserr - 0.5 * delta)

@jax.jit
def cgradient_huber_loss(y_true, raw_prediction, delta):
    res = raw_prediction - y_true
    if jnp.abs(res) <= delta:
        return res
    else:
        return delta if res >= 0 else -delta

@jax.jit
def cgrad_hess_huber_loss(y_true, raw_prediction, delta):
    gh = jnp.zeros(2)
    gh[1] = raw_prediction - y_true
    if jnp.abs(gh[1]) <= delta:
        gh[0] = gh[1]
        gh[1] = 1.0
    else:
        gh[0] = delta if gh[1] >= 0 else -delta
        gh[1] = 0.0
    return gh

@jax.jit
def closs_half_poisson(y_true, raw_prediction):
    return jnp.exp(raw_prediction) - y_true * raw_prediction

@jax.jit
def cgradient_half_poisson(y_true, raw_prediction):
    return jnp.exp(raw_prediction) - y_true

@jax.jit
def closs_grad_half_poisson(y_true, raw_prediction):
    lg = jnp.zeros(2)
    lg[1] = jnp.exp(raw_prediction)
    lg[0] = lg[1] - y_true * raw_prediction
    lg[1] -= y_true
    return lg

@jax.jit
def cgrad_hess_half_poisson(y_true, raw_prediction):
    gh = jnp.zeros(2)
    gh[1] = jnp.exp(raw_prediction)
    gh[0] = gh[1] - y_true
    return gh

@jax.jit
def closs_half_gamma(y_true, raw_prediction):
    return raw_prediction + y_true * jnp.exp(-raw_prediction)

@jax.jit
def cgradient_half_gamma(y_true, raw_prediction):
    return 1. - y_true * jnp.exp(-raw_prediction)

@jax.jit
def closs_grad_half_gamma(y_true, raw_prediction):
    lg = jnp.zeros(2)
    lg[1] = jnp.exp(-raw_prediction)
    lg[0] = raw_prediction + y_true * lg[1]
    lg[1] = 1. - y_true * lg[1]
    return lg

@jax.jit
def cgrad_hess_half_gamma(y_true, raw_prediction):
    gh = jnp.zeros(2)
    gh[1] = jnp.exp(-raw_prediction)
    gh[0] = 1. - y_true * gh[1]
    gh[1] *= y_true
    return gh

@jax.jit
def closs_half_tweedie(y_true, raw_prediction, power):
    if power == 0.:
        return closs_half_squared_error(y_true, jnp.exp(raw_prediction))
    elif power == 1.:
        return closs_half_poisson(y_true, raw_prediction)
    elif power == 2.:
        return closs_half_gamma(y_true, raw_prediction)
    else:
        return (jnp.exp((2. - power) * raw_prediction) / (2. - power)
                - y_true * jnp.exp((1. - power) * raw_prediction) / (1. - power))

@jax.jit
def cgradient_half_tweedie(y_true, raw_prediction, power):
    if power == 0.:
        exp1 = jnp.exp(raw_prediction)
        return exp1 * (exp1 - y_true)
    elif power == 1.:
        return cgradient_half_poisson(y_true, raw_prediction)
    elif power == 2.:
        return cgradient_half_gamma(y_true, raw_prediction)
    else:
        return (jnp.exp((2. - power) * raw_prediction)
                - y_true * jnp.exp((1. - power) * raw_prediction))

@jax.jit
def closs_grad_half_tweedie(y_true, raw_prediction, power):
    lg = jnp.zeros(2)
    if power == 0.:
        exp1 = jnp.exp(raw_prediction)
        lg[0] = closs_half_squared_error(y_true, exp1)
        lg[1] = exp1 * (exp1 - y_true)
    elif power == 1.:
        return closs_grad_half_poisson(y_true, raw_prediction)
    elif power == 2.:
        return closs_grad_half_gamma(y_true, raw_prediction)
    else:
        exp1 = jnp.exp((1. - power) * raw_prediction)
        exp2 = jnp.exp((2. - power) * raw_prediction)
        lg[0] = exp2 / (2. - power) - y_true * exp1 / (1. - power)
        lg[1] = exp2 - y_true * exp1
    return lg

@jax.jit
def cgrad_hess_half_tweedie(y_true, raw_prediction, power):
    gh = jnp.zeros(2)
    if power == 0.:
        exp1 = jnp.exp(raw_prediction)
        gh[0] = exp1 * (exp1 - y_true)
        gh[1] = exp1 * (2 * exp1 - y_true)
    elif power == 1.:
        return cgrad_hess_half_poisson(y_true, raw_prediction)
    elif power == 2.:
        return cgrad_hess_half_gamma(y_true, raw_prediction)
    else:
        exp1 = jnp.exp((1. - power) * raw_prediction)
        exp2 = jnp.exp((2. - power) * raw_prediction)
        gh[0] = exp2 - y_true * exp1
        gh[1] = (2. - power) * exp2 - (1. - power) * y_true * exp1
    return gh

@jax.jit
def closs_half_tweedie_identity(y_true, raw_prediction, power):
    if power == 0.:
        return closs_half_squared_error(y_true, raw_prediction)
    elif power == 1.:
        if y_true == 0:
            return raw_prediction
        else:
            return y_true * jnp.log(y_true/raw_prediction) + raw_prediction - y_true
    elif power == 2.:
        return jnp.log(raw_prediction/y_true) + y_true/raw_prediction - 1.
    else:
        tmp = jnp.power(raw_prediction, 1. - power)
        tmp = raw_prediction * tmp / (2. - power) - y_true * tmp / (1. - power)
        if y_true > 0:
            tmp += jnp.power(y_true, 2. - power) / ((1. - power) * (2. - power))
        return tmp

@jax.jit
def cgradient_half_tweedie_identity(y_true, raw_prediction, power):
    if power == 0.:
        return raw_prediction - y_true
    elif power == 1.:
        return 1. - y_true / raw_prediction
    elif power == 2.:
        return (raw_prediction - y_true) / (raw_prediction * raw_prediction)
    else:
        return jnp.power(raw_prediction, -power) * (raw_prediction - y_true)

@jax.jit
def closs_grad_half_tweedie_identity(y_true, raw_prediction, power):
    lg = jnp.zeros(2)
    tmp = jnp.power(raw_prediction, 1. - power)
    if power == 0.:
        lg[1] = raw_prediction - y_true
        lg[0] = 0.5 * lg[1] * lg[1]
    elif power == 1.:
        if y_true == 0:
            lg[0] = raw_prediction
        else:
            lg[0] = (y_true * jnp.log(y_true/raw_prediction)
                     + raw_prediction - y_true)
        lg[1] = 1. - y_true / raw_prediction
    elif power == 2.:
        lg[0] = (jnp.log(raw_prediction/y_true)
                 + y_true/raw_prediction - 1.)
        tmp = raw_prediction * raw_prediction
        lg[1] = (raw_prediction - y_true) / tmp
    else:
        tmp = jnp.power(raw_prediction, 1. - power)
        lg[0] = (raw_prediction * tmp / (2. - power)
                 - y_true * tmp / (1. - power))
        if y_true > 0:
            lg[0] += (jnp.power(y_true, 2. - power)
                      / ((1. - power) * (2. - power)))
        lg[1] = tmp * (1. - y_true / raw_prediction)
    return lg

@jax.jit
def cgrad_hess_half_tweedie_identity(y_true, raw_prediction, power):
    gh = jnp.zeros(2)
    tmp = jnp.power(raw_prediction, -power)
    if power == 0.:
        gh[0] = raw_prediction - y_true
        gh[1] = 1.0
    elif power == 1.:
        gh[0] = 1. - y_true / raw_prediction
        gh[1] = y_true / (raw_prediction * raw_prediction)
    elif power == 2.:
        tmp = raw_prediction * raw_prediction
        gh[0] = (raw_prediction - y_true) / tmp
        gh[1] = (-1. + 2. * y_true / raw_prediction) / tmp
    else:
        tmp = jnp.power(raw_prediction, -power)
        gh[0] = tmp * (raw_prediction - y_true)
        gh[1] = tmp * ((1. - power) + power * y_true / raw_prediction)
    return gh

@jax.jit
def closs_half_binomial(y_true, raw_prediction):
    return jnp.log1p(jnp.exp(raw_prediction)) - y_true * raw_prediction

@jax.jit
def cgradient_half_binomial(y_true, raw_prediction):
    exp_tmp = jnp.exp(-raw_prediction)
    return ((1 - y_true) - y_true * exp_tmp) / (1 + exp_tmp)

@jax.jit
def closs_grad_half_binomial(y_true, raw_prediction):
    lg = jnp.zeros(2)
    if raw_prediction <= 0:
        lg[1] = jnp.exp(raw_prediction)
        if raw_prediction <= -37:
            lg[0] = lg[1] - y_true * raw_prediction
        else:
            lg[0] = jnp.log1p(lg[1]) - y_true * raw_prediction
        lg[1] = ((1 - y_true) * lg[1] - y_true) / (1 + lg[1])
    else:
        lg[1] = jnp.exp(-raw_prediction)
        if raw_prediction <= 18:
            lg[0] = jnp.log1p(lg[1]) + (1 - y_true) * raw_prediction
        else:
            lg[0] = lg[1] + (1 - y_true) * raw_prediction
        lg[1] = ((1 - y_true) - y_true * lg[1]) / (1 + lg[1])
    return lg

@jax.jit
def cgrad_hess_half_binomial(y_true, raw_prediction):
    gh = jnp.zeros(2)
    gh[1] = jnp.exp(-raw_prediction)
    gh[0] = ((1 - y_true) - y_true * gh[1]) / (1 + gh[1])
    gh[1] = gh[1] / (1 + gh[1])**2
    return gh

@jax.jit
def closs_exponential(y_true, raw_prediction):
    tmp = jnp.exp(raw_prediction)
    return y_true / tmp + (1 - y_true) * tmp

@jax.jit
def cgradient_exponential(y_true, raw_prediction):
    tmp = jnp.exp(raw_prediction)
    return -y_true / tmp + (1 - y_true) * tmp

@jax.jit
def closs_grad_exponential(y_true, raw_prediction):
    lg = jnp.zeros(2)
    lg[1] = jnp.exp(raw_prediction)
    lg[0] = y_true / lg[1] + (1 - y_true) * lg[1]
    lg[1] = -y_true / lg[1] + (1 - y_true) * lg[1]
    return lg

@jax.jit
def cgrad_hess_exponential(y_true, raw_prediction):
    gh = jnp.zeros(2)
    gh[1] = jnp.exp(raw_prediction)
    gh[0] = -y_true / gh[1] + (1 - y_true) * gh[1]
    gh[1] = y_true / gh[1] + (1 - y_true) * gh[1]
    return gh




class PyLossFunction:
    """Base class for convex loss functions."""

    def loss(self, y_true, raw_prediction, sample_weight=None):
        """Compute the pointwise loss value for each input.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array-like of shape (n_samples,)
            Raw prediction values (in link space).
        sample_weight : array-like of shape (n_samples,) or None, optional
            Sample weights.

        Returns
        -------
        loss : array-like of shape (n_samples,)
            Element-wise loss function.
        """
        pass

    def gradient(self, y_true, raw_prediction, sample_weight=None):
        """Compute gradient of loss w.r.t raw_prediction for each input.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array-like of shape (n_samples,)
            Raw prediction values (in link space).
        sample_weight : array-like of shape (n_samples,) or None, optional
            Sample weights.

        Returns
        -------
        gradient : array-like of shape (n_samples,)
            Element-wise gradients.
        """
        pass

    def loss_gradient(self, y_true, raw_prediction, sample_weight=None):
        """Compute loss and gradient of loss w.r.t raw_prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array-like of shape (n_samples,)
            Raw prediction values (in link space).
        sample_weight : array-like of shape (n_samples,) or None, optional
            Sample weights.

        Returns
        -------
        loss : array-like of shape (n_samples,)
            Element-wise loss function.

        gradient : array-like of shape (n_samples,)
            Element-wise gradients.
        """
        loss = self.loss(y_true, raw_prediction, sample_weight)
        gradient = self.gradient(y_true, raw_prediction, sample_weight)
        return jnp.asarray(loss), jnp.asarray(gradient)

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None):
        """Compute gradient and hessian of loss w.r.t raw_prediction.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array-like of shape (n_samples,)
            Raw prediction values (in link space).
        sample_weight : array-like of shape (n_samples,) or None, optional
            Sample weights.

        Returns
        -------
        gradient : array-like of shape (n_samples,)
            Element-wise gradients.

        hessian : array-like of shape (n_samples,)
            Element-wise hessians.
        """
        pass

    @jax.jit
    def _loss(self, y_true, raw_prediction):
        """Compute the loss for a single sample.

        Parameters
        ----------
        y_true : float
            Observed, true target value.
        raw_prediction : float
            Raw prediction value (in link space).

        Returns
        -------
        float
            The loss evaluated at `y_true` and `raw_prediction`.
        """
        pass

    @jax.jit
    def _gradient(self, y_true, raw_prediction):
        """Compute gradient of loss w.r.t. raw_prediction for a single sample.

        Parameters
        ----------
        y_true : float
            Observed, true target value.
        raw_prediction : float
            Raw prediction value (in link space).

        Returns
        -------
        float
            The derivative of the loss function w.r.t. `raw_prediction`.
        """
        pass

    @jax.jit
    def _grad_hess(self, y_true, raw_prediction):
        """Compute gradient and hessian.

        Gradient and hessian of loss w.r.t. raw_prediction for a single sample.

        This is usually diagonal in raw_prediction_i and raw_prediction_j.
        Therefore, we return the diagonal element i=j.

        For a loss with a non-canonical link, this might implement the diagonal
        of the Fisher matrix (=expected hessian) instead of the hessian.

        Parameters
        ----------
        y_true : float
            Observed, true target value.
        raw_prediction : float
            Raw prediction value (in link space).

        Returns
        -------
        (float, float)
            Gradient and hessian of the loss function w.r.t. `raw_prediction`.
        """
        pass



class PyHalfSquaredError(PyLossFunction):
    """Half Squared Error with identity link.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction
    """

    def _loss(self, y_true, raw_prediction):
        return self.closs_half_squared_error(y_true, raw_prediction)

    def _gradient(self, y_true, raw_prediction):
        return self.cgradient_half_squared_error(y_true, raw_prediction)

    def _grad_hess(self, y_true, raw_prediction):
        return self.cgrad_hess_half_squared_error(y_true, raw_prediction)

    def loss(
        self,
        y_true,          # IN
        raw_prediction,  # IN
        sample_weight=None,   # IN
        n_threads=1
    ):
        n_samples = y_true.shape[0]

        if sample_weight is None:
            loss_out = jax.vmap(self.closs_half_squared_error)(y_true, raw_prediction)
        else:
            loss_out = jax.vmap(lambda y, p, w: w * self.closs_half_squared_error(y, p))(y_true, raw_prediction, sample_weight)

        return jnp.asarray(loss_out)


    def gradient(
        self,
        y_true,          # IN
        raw_prediction,  # IN
        sample_weight=None,   # IN
        n_threads=1
    ):
        n_samples = y_true.shape[0]

        if sample_weight is None:
            gradient_out = jax.vmap(self.cgradient_half_squared_error)(y_true, raw_prediction)
        else:
            gradient_out = jax.vmap(lambda y, p, w: w * self.cgradient_half_squared_error(y, p))(y_true, raw_prediction, sample_weight)

        return jnp.asarray(gradient_out)

    def gradient_hessian(
        self,
        y_true,          # IN
        raw_prediction,  # IN
        sample_weight=None,   # IN
        n_threads=1
    ):
        n_samples = y_true.shape[0]

        if sample_weight is None:
            grad_hess_out = jax.vmap(self.cgrad_hess_half_squared_error)(y_true, raw_prediction)
        else:
            grad_hess_out = jax.vmap(lambda y, p, w: (w * self.cgrad_hess_half_squared_error(y, p).val1, w * self.cgrad_hess_half_squared_error(y, p).val2))(y_true, raw_prediction, sample_weight)

        gradient_out, hessian_out = zip(*grad_hess_out)

        return jnp.asarray(gradient_out), jnp.asarray(hessian_out)



class PyAbsoluteError(PyLossFunction):
    """Absolute Error with identity link.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction
    """

    def _loss(self, y_true, raw_prediction):
        return closs_absolute_error(y_true, raw_prediction)

    def _gradient(self, y_true, raw_prediction):
        return cgradient_absolute_error(y_true, raw_prediction)

    def _grad_hess(self, y_true, raw_prediction):
        return cgrad_hess_absolute_error(y_true, raw_prediction)

    def loss(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        loss_out = jnp.zeros_like(y_true)

        if sample_weight is None:
            for i in jax.prange(n_samples, static_schedule=True, num_threads=n_threads):
                loss_out[i] = closs_absolute_error(y_true[i], raw_prediction[i])
        else:
            for i in jax.prange(n_samples, static_schedule=True, num_threads=n_threads):
                loss_out[i] = sample_weight[i] * closs_absolute_error(y_true[i], raw_prediction[i])

        return loss_out

    def gradient(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        gradient_out = jnp.zeros_like(y_true)

        if sample_weight is None:
            for i in jax.prange(n_samples, static_schedule=True, num_threads=n_threads):
                gradient_out[i] = cgradient_absolute_error(y_true[i], raw_prediction[i])
        else:
            for i in jax.prange(n_samples, static_schedule=True, num_threads=n_threads):
                gradient_out[i] = sample_weight[i] * cgradient_absolute_error(y_true[i], raw_prediction[i])

        return gradient_out

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        gradient_out = jnp.zeros_like(y_true)
        hessian_out = jnp.zeros_like(y_true)
        dbl2 = jnp.zeros((2,))

        if sample_weight is None:
            for i in jax.prange(n_samples, static_schedule=True, num_threads=n_threads):
                dbl2 = cgrad_hess_absolute_error(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2[0]
                hessian_out[i] = dbl2[1]
        else:
            for i in jax.prange(n_samples, static_schedule=True, num_threads=n_threads):
                dbl2 = cgrad_hess_absolute_error(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2[0]
                hessian_out[i] = sample_weight[i] * dbl2[1]

        return gradient_out, hessian_out



class PyPinballLoss(PyLossFunction):
    """Quantile Loss aka Pinball Loss with identity link.

    Domain:
    y_true and y_pred all real numbers
    quantile in (0, 1)

    Link:
    y_pred = raw_prediction

    Note: 2 * PyPinballLoss(quantile=0.5) equals PyAbsoluteError()
    """

    def __init__(self, quantile):
        self.quantile = quantile

    def py_loss(self, y_true, raw_prediction):
        return jax.vmap(closs_pinball_loss)(y_true, raw_prediction, self.quantile)

    def py_gradient(self, y_true, raw_prediction):
        return jax.vmap(cgradient_pinball_loss)(y_true, raw_prediction, self.quantile)

    def py_grad_hess(self, y_true, raw_prediction):
        return jax.vmap(cgrad_hess_pinball_loss)(y_true, raw_prediction, self.quantile)

    def loss(self, y_true, raw_prediction, sample_weight=None):
        if sample_weight is None:
            return self.py_loss(y_true, raw_prediction)
        else:
            return sample_weight * self.py_loss(y_true, raw_prediction)

    def gradient(self, y_true, raw_prediction, sample_weight=None):
        if sample_weight is None:
            return self.py_gradient(y_true, raw_prediction)
        else:
            return sample_weight * self.py_gradient(y_true, raw_prediction)

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None):
        gradient_out, hessian_out = self.py_grad_hess(y_true, raw_prediction)
        if sample_weight is None:
            return gradient_out, hessian_out
        else:
            return sample_weight * gradient_out, sample_weight * hessian_out


import jax
import jax.numpy as np

class PyHuberLoss(PyLossFunction):
    """Huber Loss with identity link.

    Domain:
    y_true and y_pred all real numbers
    delta in positive real numbers

    Link:
    y_pred = raw_prediction
    """

    def __init__(self, delta):
        self.delta = delta

    def py_loss(self, y_true, raw_prediction):
        return jax.vmap(closs_huber_loss)(y_true, raw_prediction, self.delta)

    def py_gradient(self, y_true, raw_prediction):
        return jax.vmap(cgradient_huber_loss)(y_true, raw_prediction, self.delta)

    def py_grad_hess(self, y_true, raw_prediction):
        return jax.vmap(cgrad_hess_huber_loss)(y_true, raw_prediction, self.delta)

    def loss(self, y_true, raw_prediction, sample_weight=None):
        if sample_weight is None:
            return self.py_loss(y_true, raw_prediction)
        else:
            return sample_weight * self.py_loss(y_true, raw_prediction)

    def gradient(self, y_true, raw_prediction, sample_weight=None):
        if sample_weight is None:
            return self.py_gradient(y_true, raw_prediction)
        else:
            return sample_weight * self.py_gradient(y_true, raw_prediction)

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None):
        gradient_out, hessian_out = self.py_grad_hess(y_true, raw_prediction)
        if sample_weight is None:
            return gradient_out, hessian_out
        else:
            return sample_weight * gradient_out, sample_weight * hessian_out



class PyHalfPoissonLoss(PyLossFunction):
    """Half Poisson deviance loss with log-link.

    Domain:
    y_true in non-negative real numbers
    y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    Half Poisson deviance with log-link is
        y_true * log(y_true/y_pred) + y_pred - y_true
        = y_true * log(y_true) - y_true * raw_prediction
          + exp(raw_prediction) - y_true

    Dropping constant terms, this gives:
        exp(raw_prediction) - y_true * raw_prediction
    """

    def __init__(self):
        super().__init__()

    def loss(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        def loss_fn(y_true, raw_prediction):
            return jnp.exp(raw_prediction) - y_true * raw_prediction

        loss_fn = jax.jit(loss_fn)

        if sample_weight is None:
            loss_out = jax.vmap(loss_fn)(y_true, raw_prediction)
        else:
            loss_out = jax.vmap(lambda y, p, w: w * loss_fn(y, p))(y_true, raw_prediction, sample_weight)

        return loss_out

    def loss_gradient(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        def loss_gradient_fn(y_true, raw_prediction):
            loss = jnp.exp(raw_prediction) - y_true * raw_prediction
            gradient = jnp.exp(raw_prediction) - y_true
            return loss, gradient

        loss_gradient_fn = jax.jit(loss_gradient_fn)

        if sample_weight is None:
            loss_out, gradient_out = jax.vmap(loss_gradient_fn)(y_true, raw_prediction)
        else:
            loss_out, gradient_out = jax.vmap(lambda y, p, w: (w * loss_fn(y, p), w * gradient_fn(y, p)))(y_true, raw_prediction, sample_weight)

        return loss_out, gradient_out

    def gradient(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        def gradient_fn(y_true, raw_prediction):
            return jnp.exp(raw_prediction) - y_true

        gradient_fn = jax.jit(gradient_fn)

        if sample_weight is None:
            gradient_out = jax.vmap(gradient_fn)(y_true, raw_prediction)
        else:
            gradient_out = jax.vmap(lambda y, p, w: w * gradient_fn(y, p))(y_true, raw_prediction, sample_weight)

        return gradient_out

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        def gradient_hessian_fn(y_true, raw_prediction):
            gradient = jnp.exp(raw_prediction) - y_true
            hessian = jnp.exp(raw_prediction)
            return gradient, hessian

        gradient_hessian_fn = jax.jit(gradient_hessian_fn)

        if sample_weight is None:
            gradient_out, hessian_out = jax.vmap(gradient_hessian_fn)(y_true, raw_prediction)
        else:
            gradient_out, hessian_out = jax.vmap(lambda y, p, w: (w * gradient_fn(y, p), w * hessian_fn(y, p)))(y_true, raw_prediction, sample_weight)

        return gradient_out, hessian_out



class PyHalfGammaLoss(PyLossFunction):
    """Half Gamma deviance loss with log-link.

    Domain:
    y_true and y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    Half Gamma deviance with log-link is
        log(y_pred/y_true) + y_true/y_pred - 1
        = raw_prediction - log(y_true) + y_true * exp(-raw_prediction) - 1

    Dropping constant terms, this gives:
        raw_prediction + y_true * exp(-raw_prediction)
    """

    def _loss(self, y_true, raw_prediction):
        return closs_half_gamma(y_true, raw_prediction)

    def _gradient(self, y_true, raw_prediction):
        return cgradient_half_gamma(y_true, raw_prediction)

    def _grad_hess(self, y_true, raw_prediction):
        return cgrad_hess_half_gamma(y_true, raw_prediction)

    def loss(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        loss_out = jnp.zeros_like(y_true, dtype=np.float64)

        if sample_weight is None:
            for i in range(n_samples):
                loss_out[i] = closs_half_gamma(y_true[i], raw_prediction[i])
        else:
            for i in range(n_samples):
                loss_out[i] = sample_weight[i] * closs_half_gamma(y_true[i], raw_prediction[i])

        return loss_out

    def loss_gradient(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        loss_out = jnp.zeros_like(y_true, dtype=np.float64)
        gradient_out = jnp.zeros_like(y_true, dtype=np.float64)

        if sample_weight is None:
            for i in range(n_samples):
                dbl2 = closs_grad_half_gamma(y_true[i], raw_prediction[i])
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in range(n_samples):
                dbl2 = closs_grad_half_gamma(y_true[i], raw_prediction[i])
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return loss_out, gradient_out

    def gradient(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        gradient_out = jnp.zeros_like(y_true, dtype=np.float64)

        if sample_weight is None:
            for i in range(n_samples):
                gradient_out[i] = cgradient_half_gamma(y_true[i], raw_prediction[i])
        else:
            for i in range(n_samples):
                gradient_out[i] = sample_weight[i] * cgradient_half_gamma(y_true[i], raw_prediction[i])

        return gradient_out

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        gradient_out = jnp.zeros_like(y_true, dtype=np.float64)
        hessian_out = jnp.zeros_like(y_true, dtype=np.float64)

        if sample_weight is None:
            for i in range(n_samples):
                dbl2 = cgrad_hess_half_gamma(y_true[i], raw_prediction[i])
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in range(n_samples):
                dbl2 = cgrad_hess_half_gamma(y_true[i], raw_prediction[i])
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return gradient_out, hessian_out


class PyHalfTweedieLoss(PyLossFunction):
    """Half Tweedie deviance loss with log-link.

    Domain:
    y_true in real numbers if p <= 0
    y_true in non-negative real numbers if 0 < p < 2
    y_true in positive real numbers if p >= 2
    y_pred and power in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    Half Tweedie deviance with log-link and p=power is
        max(y_true, 0)**(2-p) / (1-p) / (2-p)
        - y_true * y_pred**(1-p) / (1-p)
        + y_pred**(2-p) / (2-p)
        = max(y_true, 0)**(2-p) / (1-p) / (2-p)
        - y_true * exp((1-p) * raw_prediction) / (1-p)
        + exp((2-p) * raw_prediction) / (2-p)

    Dropping constant terms, this gives:
        exp((2-p) * raw_prediction) / (2-p)
        - y_true * exp((1-p) * raw_prediction) / (1-p)

    Notes:
    - Poisson with p=1 and and Gamma with p=2 have different terms dropped such
      that cHalfTweedieLoss is not continuous in p=power at p=1 and p=2.
    - While the Tweedie distribution only exists for p<=0 or p>=1, the range
      0<p<1 still gives a strictly consistent scoring function for the
      expectation.
    """

    def __init__(self, power):
        self.power = power

    def loss(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        loss_out = jnp.empty_like(y_true)

        if sample_weight is None:
            for i in range(n_samples):
                loss_out[i] = closs_half_tweedie(y_true[i], raw_prediction[i], self.power)
        else:
            for i in range(n_samples):
                loss_out[i] = sample_weight[i] * closs_half_tweedie(y_true[i], raw_prediction[i], self.power)

        return loss_out

    def loss_gradient(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        loss_out = jnp.empty_like(y_true)
        gradient_out = jnp.empty_like(y_true)
        dbl2 =  DoublePair()

        if sample_weight is None:
            for i in range(n_samples):
                dbl2 = closs_grad_half_tweedie(y_true[i], raw_prediction[i], self.power)
                loss_out[i] = dbl2.val1
                gradient_out[i] = dbl2.val2
        else:
            for i in range(n_samples):
                dbl2 = closs_grad_half_tweedie(y_true[i], raw_prediction[i], self.power)
                loss_out[i] = sample_weight[i] * dbl2.val1
                gradient_out[i] = sample_weight[i] * dbl2.val2

        return loss_out, gradient_out

    def gradient(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        gradient_out = jnp.empty_like(y_true)

        if sample_weight is None:
            for i in range(n_samples):
                gradient_out[i] = cgradient_half_tweedie(y_true[i], raw_prediction[i], self.power)
        else:
            for i in range(n_samples):
                gradient_out[i] = sample_weight[i] * cgradient_half_tweedie(y_true[i], raw_prediction[i], self.power)

        return gradient_out

    def gradient_hessian(self, y_true, raw_prediction, sample_weight=None, n_threads=1):
        n_samples = y_true.shape[0]
        gradient_out = jnp.empty_like(y_true)
        hessian_out = jnp.empty_like(y_true)
        dbl2 =  DoublePair()

        if sample_weight is None:
            for i in range(n_samples):
                dbl2 = cgrad_hess_half_tweedie(y_true[i], raw_prediction[i], self.power)
                gradient_out[i] = dbl2.val1
                hessian_out[i] = dbl2.val2
        else:
            for i in range(n_samples):
                dbl2 = cgrad_hess_half_tweedie(y_true[i], raw_prediction[i], self.power)
                gradient_out[i] = sample_weight[i] * dbl2.val1
                hessian_out[i] = sample_weight[i] * dbl2.val2

        return gradient_out, hessian_out