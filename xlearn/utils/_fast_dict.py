import jax.numpy as jnp

class IntFloatDict:
    def __init__(self, keys, values):
        self.my_dict = dict(zip(keys, values))

    def __len__(self):
        return len(self.my_dict)

    def __getitem__(self, key):
        return self.my_dict[key]

    def __setitem__(self, key, value):
        self.my_dict[key] = value

    def __iter__(self):
        for key, value in self.my_dict.items():
            yield key, value

    def to_arrays(self):
        keys = jnp.array(list(self.my_dict.keys()))
        values = jnp.array(list(self.my_dict.values()))
        return keys, values

    def update(self, other):
        self.my_dict.update(other.my_dict)

    def copy(self):
        new_dict = IntFloatDict([], [])
        new_dict.my_dict = self.my_dict.copy()
        return new_dict

    def append(self, key, value):
        self.my_dict[key] = value

def argmin(d):
    min_key = min(d.my_dict, key=d.my_dict.get)
    min_value = d.my_dict[min_key]
    return min_key, min_value
