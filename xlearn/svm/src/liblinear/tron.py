import math
import jax.numpy as jnp

class TRON:
    def __init__(self, fun_obj, eps, max_iter, blas):
        self.fun_obj = fun_obj
        self.eps = eps
        self.max_iter = max_iter
        self.blas = blas
        self.tron_print_string = self.default_print

    def default_print(self, buf):
        print(buf)

    def info(self, fmt, *args):
        buf = fmt % args
        self.tron_print_string(buf)

    def tron(self, w):
        eta0 = 1e-4
        eta1 = 0.25
        eta2 = 0.75
        sigma1 = 0.25
        sigma2 = 0.5
        sigma3 = 4

        n = self.fun_obj.get_nr_variable()
        cg_iter = 0
        delta = 0.0
        snorm = 0.0
        alpha = 0.0
        f = 0.0
        fnew = 0.0
        prered = 0.0
        actred = 0.0
        gs = 0.0
        search = True
        iter = 1
        inc = 1
        s = jnp.zeros(n)
        r = jnp.zeros(n)
        w_new = jnp.zeros(n)
        g = jnp.zeros(n)

        for i in range(n):
            w[i] = 0.0

        f = self.fun_obj.fun(w)
        self.fun_obj.grad(w, g)
        delta = jnp.linalg.norm(g)
        gnorm1 = delta
        gnorm = gnorm1

        if gnorm <= self.eps * gnorm1:
            search = False

        while iter <= self.max_iter and search:
            cg_iter = self.trcg(delta, g, s, r)

            w_new = w.copy()
            w_new += s

            gs = jnp.dot(g, s)
            prered = -0.5 * (gs - jnp.dot(s, r))
            fnew = self.fun_obj.fun(w_new)

            actred = f - fnew

            snorm = jnp.linalg.norm(s)
            if iter == 1:
                delta = min(delta, snorm)

            if fnew - f - gs <= 0:
                alpha = sigma3
            else:
                alpha = max(sigma1, -0.5 * (gs / (fnew - f - gs)))

            if actred < eta0 * prered:
                delta = min(max(alpha, sigma1) * snorm, sigma2 * delta)
            elif actred < eta1 * prered:
                delta = max(sigma1 * delta, min(alpha * snorm, sigma2 * delta))
            elif actred < eta2 * prered:
                delta = max(sigma1 * delta, min(alpha * snorm, sigma3 * delta))
            else:
                delta = max(delta, min(alpha * snorm, sigma3 * delta))

            self.info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter)

            if actred > eta0 * prered:
                iter += 1
                w = w_new.copy()
                f = fnew
                self.fun_obj.grad(w, g)

                gnorm = jnp.linalg.norm(g)
                if gnorm <= self.eps * gnorm1:
                    break

            if f < -1.0e+32:
                self.info("WARNING: f < -1.0e+32\n")
                break

            if abs(actred) <= 0 and prered <= 0:
                self.info("WARNING: actred and prered <= 0\n")
                break

            if abs(actred) <= 1.0e-12 * abs(f) and abs(prered) <= 1.0e-12 * abs(f):
                self.info("WARNING: actred and prered too small\n")
                break

        return iter - 1

    def trcg(self, delta, g, s, r):
        n = self.fun_obj.get_nr_variable()
        d = jnp.zeros(n)
        Hd = jnp.zeros(n)
        rTr = 0.0
        rnewTrnew = 0.0
        alpha = 0.0
        beta = 0.0
        cgtol = 0.0

        for i in range(n):
            s[i] = 0.0
            r[i] = -g[i]
            d[i] = r[i]

        cgtol = 0.1 * jnp.linalg.norm(g)

        cg_iter = 0
        rTr = jnp.dot(r, r)

        while True:
            if jnp.linalg.norm(r) <= cgtol:
                break

            cg_iter += 1
            self.fun_obj.Hv(d, Hd)

            alpha = rTr / jnp.dot(d, Hd)
            s += alpha * d

            if jnp.linalg.norm(s) > delta:
                self.info("cg reaches trust region boundary\n")
                alpha = -alpha
                s += alpha * d

                std = jnp.dot(s, d)
                sts = jnp.dot(s, s)
                dtd = jnp.dot(d, d)
                dsq = delta * delta
                rad = math.sqrt(std * std + dtd * (dsq - sts))

                if std >= 0:
                    alpha = (dsq - sts) / (std + rad)
                else:
                    alpha = (rad - std) / dtd

                s += alpha * d
                alpha = -alpha
                r += alpha * Hd
                break

            alpha = -alpha
            r += alpha * Hd
            rnewTrnew = jnp.dot(r, r)
            beta = rnewTrnew / rTr
            d *= beta
            d += r
            rTr = rnewTrnew

        return cg_iter

    def norm_inf(self, x):
        return jnp.max(jnp.abs(x))

    def set_print_string(self, print_string):
        self.tron_print_string = print_string
