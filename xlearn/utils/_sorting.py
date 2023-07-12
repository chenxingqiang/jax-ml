import jax.numpy as jnp

def dual_swap(darr, iarr, a, b):
    """Swap the values at index a and b of both darr and iarr"""
    darr[a], darr[b] = darr[b], darr[a]
    iarr[a], iarr[b] = iarr[b], iarr[a]

def simultaneous_sort(values, indices):
    """
    Perform a recursive quicksort on the values array as to sort them ascendingly.
    This simultaneously performs the swaps on both the values and the indices arrays.
    """
    size = len(values)

    if size <= 1:
        return values, indices
    elif size == 2:
        if values[0] > values[1]:
            dual_swap(values, indices, 0, 1)
    elif size == 3:
        if values[0] > values[1]:
            dual_swap(values, indices, 0, 1)
        if values[1] > values[2]:
            dual_swap(values, indices, 1, 2)
            if values[0] > values[1]:
                dual_swap(values, indices, 0, 1)
    else:
        pivot_idx = size // 2
        if values[0] > values[size - 1]:
            dual_swap(values, indices, 0, size - 1)
        if values[size - 1] > values[pivot_idx]:
            dual_swap(values, indices, size - 1, pivot_idx)
            if values[0] > values[size - 1]:
                dual_swap(values, indices, 0, size - 1)
        pivot_val = values[size - 1]

        store_idx = 0
        for i in range(size - 1):
            if values[i] < pivot_val:
                dual_swap(values, indices, i, store_idx)
                store_idx += 1
        dual_swap(values, indices, store_idx, size - 1)
        pivot_idx = store_idx

        if pivot_idx > 1:
            simultaneous_sort(values[:pivot_idx], indices[:pivot_idx])
        if pivot_idx + 2 < size:
            simultaneous_sort(values[pivot_idx + 1:], indices[pivot_idx + 1:])
    
    return values, indices
