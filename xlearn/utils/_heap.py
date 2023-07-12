import heapq

def heap_push(values, indices, size, val, val_idx):
    """Push a tuple (val, val_idx) onto a fixed-size max-heap.

    The max-heap is represented as a Structure of Arrays where:
     - values is the array containing the data to construct the heap with
     - indices is the array containing the indices (meta-data) of each value

    Notes
    -----
    This function modifies the input lists in-place.

    For instance:

        values = [1.2, 0.4, 0.1],
        indices = [42, 1, 5],
        heap_push(
            values=values,
            indices=indices,
            size=3,
            val=0.2,
            val_idx=4,
        )

    will modify values and indices inplace, giving at the end of the call:

        values  == [0.4, 0.2, 0.1]
        indices == [1, 4, 5]

    """
    # Check if val should be in heap
    if val >= -values[0]:
        return

    # Python's heapq library only provides a min-heap, so we invert the
    # values to get the behavior of a max-heap
    heapq.heapreplace(values, -val)
    heapq.heapreplace(indices, val_idx)
