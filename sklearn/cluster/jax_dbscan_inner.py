from typing import List
from jax import numpy as jnp

def dbscan_inner(is_core: jnp.ndarray, neighborhoods: List[jnp.ndarray], labels: jnp.ndarray):
    label_num = 0

    def dfs(i, label_num):
        if labels[i] != -1 or not is_core[i]:
            return label_num

        stack = [i]
        while stack:
            i = stack[-1]
            stack = stack[:-1]

            if labels[i] == -1:
                labels = labels.at[i].set(label_num)
                if is_core[i]:
                    neighb = neighborhoods[i]
                    for v in neighb:
                        if labels[v] == -1:
                            stack.append(v)

        return label_num + 1

    for i in range(labels.shape[0]):
        label_num = dfs(i, label_num)

    return labels
