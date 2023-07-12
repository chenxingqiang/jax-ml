import jax.numpy as jnp

class BLAS_Order:
    RowMajor = 0
    ColMajor = 1

class BLAS_Trans:
    NoTrans = 110
    Trans = 116

def _dot(x, y):
    """x.T.y"""
    return jnp.vdot(x, y)

def _asum(x):
    """sum(|x_i|)"""
    return jnp.sum(jnp.abs(x))

def _axpy(alpha, x, y):
    """y := alpha * x + y"""
    return alpha * x + y

def _nrm2(x):
    """sqrt(sum((x_i)^2))"""
    return jnp.linalg.norm(x)

def _copy(x, y):
    """y := x"""
    y[:] = x

def _scal(alpha, x):
    """x := alpha * x"""
    x *= alpha

def _rotg(a, b):
    """Generate plane rotation"""
    raise NotImplementedError("Plane rotation is not supported in JAX")

def _rot(x, y, c, s):
    """Apply plane rotation"""
    raise NotImplementedError("Plane rotation is not supported in JAX")

def _gemv(ta, alpha, A, x, beta, y):
    """y := alpha * op(A).x + beta * y"""
    if ta == BLAS_Trans.Trans:
        A = A.T
    y[:] = alpha * jnp.dot(A, x) + beta * y

def _ger(alpha, x, y, A):
    """A := alpha * x.y.T + A"""
    A += alpha * jnp.outer(x, y)

def _gemm(ta, tb, alpha, A, B, beta, C):
    """C := alpha * op(A).op(B) + beta * C"""
    if ta == BLAS_Trans.Trans:
        A = A.T
    if tb == BLAS_Trans.Trans:
        B = B.T
    C[:] = alpha * jnp.dot(A, B) + beta * C
