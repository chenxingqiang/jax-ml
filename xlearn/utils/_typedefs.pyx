# _typedefs is a declaration only module
#
# The functions implemented here are for testing purpose only.


import jax.numpy as jnp


ctypedef fused testing_type_t:
    uint8_t
    intp_t
    float32_t
    float64_t
    int32_t
    int64_t


def testing_make_array_from_typed_val(testing_type_t val):
    cdef testing_type_t[:] val_view = <testing_type_t[:1]>&val
    return jnp.asarray(val_view)
