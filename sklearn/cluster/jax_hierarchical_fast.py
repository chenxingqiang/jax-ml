import jax.numpy as jnp
from typing import List, Tuple

def compute_ward_dist(m_1: jnp.ndarray, m_2: jnp.ndarray, coord_row: jnp.ndarray, coord_col: jnp.ndarray, res: jnp.ndarray) -> None:
    size_max = coord_row.shape[0]
    n_features = m_2.shape[1]
    for i in range(size_max):
        row = coord_row[i]
        col = coord_col[i]
        n = (m_1[row] * m_1[col]) / (m_1[row] + m_1[col])
        pa = jnp.sum((m_2[row, :] / m_1[row] - m_2[col, :] / m_1[col]) ** 2)
        res = jax.ops.index_update(res, i, pa * n)

def _hc_get_descendent(node: int, children: List[Tuple[int, int]], n_leaves: int) -> List[int]:
    ind = [node]
    if node < n_leaves:
        return ind
    descendent = []
    while ind:
        i = ind.pop()
        if i < n_leaves:
            descendent.append(i)
        else:
            ind.extend(children[i - n_leaves])
    return descendent

def hc_get_heads(parents: jnp.ndarray, copy: bool = True) -> jnp.ndarray:
    if copy:
        parents = parents.copy()
    size = parents.size
    for node0 in range(size - 1, -1, -1):
        node = node0
        parent = parents[node]
        while parent != node:
            parents = jax.ops.index_update(parents, node0, parent)
            node = parent
            parent = parents[node]
    return parents

def _get_parents(nodes: List[int], heads: List[int], parents: jnp.ndarray, not_visited: jnp.ndarray) -> None:
    for node in nodes:
        parent = parents[node]
        while parent != node:
            node = parent
            parent = parents[node]
        if not_visited[node]:
            not_visited = jax.ops.index_update(not_visited, node, 0)
            heads.append(node)
def max_merge(a: dict, b: dict, mask: jnp.ndarray, n_a: int, n_b: int) -> dict:
    out_obj = {}
    for key, value in a.items():
        if mask[key]:
            out_obj[key] = value

    for key, value in b.items():
        if mask[key]:
            if key in out_obj:
                out_obj[key] = max(out_obj[key], value)
            else:
                out_obj[key] = value
    return out_obj


def average_merge(a: dict, b: dict, mask: jnp.ndarray, n_a: int, n_b: int) -> dict:
    out_obj = {}
    n_out = n_a + n_b
    for key, value in a.items():
        if mask[key]:
            out_obj[key] = value

    for key, value in b.items():
        if mask[key]:
            if key in out_obj:
                out_obj[key] = (n_a * out_obj[key] + n_b * value) / n_out
            else:
                out_obj[key] = value
    return out_obj
import jax
import jax.numpy as np
from jax import jit
from flax import linen as nn

# The class WeightedEdge
class WeightedEdge:
    def __init__(self, weight, a, b):
        self.weight = weight
        self.a = a
        self.b = b

    def __lt__(self, other):
        return self.weight < other.weight

    def __le__(self, other):
        return self.weight <= other.weight

    def __eq__(self, other):
        return self.weight == other.weight

    def __ne__(self, other):
        return self.weight != other.weight

    def __gt__(self, other):
        return self.weight > other.weight

    def __ge__(self, other):
        return self.weight >= other.weight

    def __repr__(self):
        return f"{self.__class__.__name__}(weight={self.weight}, a={self.a}, b={self.b})"


# The class UnionFind
class UnionFind:
    def __init__(self, N):
        self.parent = np.full(2 * N - 1, -1, dtype=np.intp)
        self.next_label = N
        self.size = np.hstack((np.ones(N, dtype=np.intp), np.zeros(N - 1, dtype=np.intp)))

    def union(self, m, n):
        self.parent = jax.ops.index_update(self.parent, jax.ops.index[m], self.next_label)
        self.parent = jax.ops.index_update(self.parent, jax.ops.index[n], self.next_label)
        self.size = jax.ops.index_update(self.size, jax.ops.index[self.next_label], self.size[m] + self.size[n])
        self.next_label += 1

    def find(self, n):
        while self.parent[n] != -1:
            n = self.parent[n]
        return n


# The function _single_linkage_label
def _single_linkage_label(L):
    N, _ = L.shape
    result_arr = np.zeros((N, 4), dtype=np.float64)
    U = UnionFind(N + 1)

    for index in range(N):
        left, right, delta = L[index, 0], L[index, 1], L[index, 2]

        left_cluster = U.find(left)
        right_cluster = U.find(right)

        result_arr = jax.ops.index_update(result_arr, jax.ops.index[index, 0], left_cluster)
        result_arr = jax.ops.index_update(result_arr, jax.ops.index[index, 1], right_cluster)
        result_arr = jax.ops.index_update(result_arr, jax.ops.index[index, 2], delta)
        result_arr = jax.ops.index_update(result_arr, jax.ops.index[index, 3], U.size[left_cluster] + U.size[right_cluster])

        U.union(left_cluster, right_cluster)

    return result_arr
import jax
import jax.numpy as np
from jax import jit, lax

def init_union_find(N):
    parent = np.full(2 * N - 1, -1, dtype=np.intp)
    next_label = N
    size = np.hstack((np.ones(N, dtype=np.intp), np.zeros(N - 1, dtype=np.intp)))
    return parent, size, next_label

def union(parent, size, next_label, m, n):
    parent = parent.at[m, n].set(next_label)
    size = size.at[next_label].set(size[m] + size[n])
    next_label += 1
    return parent, size, next_label

def find(parent, n):
    p = n
    while parent[n] != -1:
        n = parent[n]
    parent = parent.at[p].set(n)
    return parent, n

def single_linkage_label(L):
    N, _ = L.shape
    result_arr = np.zeros((N, 4), dtype=np.float64)
    parent, size, next_label = init_union_find(N + 1)

    def body_fn(carry, x):
        result_arr, parent, size, next_label = carry
        left, right, delta = x[0], x[1], x[2]

        parent, left_cluster = find(parent, left)
        parent, right_cluster = find(parent, right)

        result_arr = result_arr.at[index, :].set([left_cluster, right_cluster, delta, size[left_cluster] + size[right_cluster]])

        parent, size, next_label = union(parent, size, next_label, left_cluster, right_cluster)

        return (result_arr, parent, size, next_label), None

    (result_arr, _, _, _), _ = lax.scan(body_fn, (result_arr, parent, size, next_label), L)

    return result_arr
import jax
import jax.numpy as np
from jax import jit, lax

@jit
def dist(raw_data, i, j):
    # You need to define the distance function here.
    # This is a placeholder that calculates Euclidean distance
    return np.sqrt(np.sum((raw_data[i] - raw_data[j])**2))

@jit
def mst_linkage_core(raw_data):
    n_samples = raw_data.shape[0]
    in_tree = np.zeros(n_samples, dtype=bool)
    result = np.zeros((n_samples - 1, 3))

    def body_fn(carry, _):
        in_tree, result, current_node, current_distances = carry
        new_distance = np.inf
        new_node = 0

        def inner_body_fn(carry, j):
            new_distance, new_node, current_distances = carry
            if in_tree[j]:
                return carry
            left_value = dist(raw_data, current
