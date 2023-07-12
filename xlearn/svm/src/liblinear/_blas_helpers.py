from typing import Callable
import jax.numpy as jnp

DotFunc = Callable[[int, jnp.ndarray, int, jnp.ndarray, int], float]
AxpyFunc = Callable[[int, float, jnp.ndarray, int, jnp.ndarray, int], None]
ScalFunc = Callable[[int, float, jnp.ndarray, int], None]
Nrm2Func = Callable[[int, jnp.ndarray, int], float]

class BlasFunctions:
    def __init__(self, dot: DotFunc, axpy: AxpyFunc, scal: ScalFunc, nrm2: Nrm2Func):
        self.dot = dot
        self.axpy = axpy
        self.scal = scal
        self.nrm2 = nrm2
