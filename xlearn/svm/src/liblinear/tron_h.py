import jax
import jax.numpy as jnp

class Function:
    def fun(self, w):
        pass

    def grad(self, w, g):
        pass

    def Hv(self, s, Hs):
        pass

    def get_nr_variable(self):
        pass

class TRON:
    def __init__(self, fun_obj, eps=0.1, max_iter=1000, blas=None):
        self.eps = eps
        self.max_iter = max_iter
        self.fun_obj = fun_obj
        self.blas = blas
        self.tron_print_string = None

    def tron(self, w):
        pass

    def set_print_string(self, i_print):
        self.tron_print_string = i_print

    def trcg(self, delta, g, s, r):
        pass

    def norm_inf(self, x):
        pass
