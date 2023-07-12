import numpy as np
from enum import Enum

class SequentialDataset64:
    """Base class for datasets with sequential data access.

    SequentialDataset is used to iterate over the rows of a matrix X and
    corresponding target values y, i.e. to iterate over samples.
    There are two methods to get the next sample:
        - next : Iterate sequentially (optionally randomized)
        - random : Iterate randomly (with replacement)

    Attributes
    ----------
    index : np.ndarray
        Index array for fast shuffling.

    current_index : int
        Index of current sample in ``index``.
        The index of current sample in the data is given by
        index[current_index].

    n_samples : int
        Number of samples in the dataset.

    seed : UINT32
        Seed used for random sampling. This attribute is modified at each call to the
        `random` method.
    """

    def next(self, x_data, x_ind, nnz, y, sample_weight):
        """Get the next example ``x`` from the dataset.

        This method gets the next sample looping sequentially over all samples.
        The order can be shuffled with the method ``shuffle``.
        Shuffling once before iterating over all samples corresponds to a
        random draw without replacement. It is used for instance in SGD solver.

        Parameters
        ----------
        x_data : list of float
            A list which holds the feature values of the next example.

        x_ind : list of int
            A list which holds the feature indices of the next example.

        nnz : list of int
            A list holding the number of non-zero values of the next example.

        y : list of float
            The target value of the next example.

        sample_weight : list of float
            The weight of the next example.
        """
        current_index = self._get_next_index()
        self._sample(x_data, x_ind, nnz, y, sample_weight, current_index)

    def random(self, x_data, x_ind, nnz, y, sample_weight):
        """Get a random example ``x`` from the dataset.

        This method gets next sample chosen randomly over a uniform
        distribution. It corresponds to a random draw with replacement.
        It is used for instance in SAG solver.

        Parameters
        ----------
        x_data : list of float
            A list which holds the feature values of the next example.

        x_ind : list of int
            A list which holds the feature indices of the next example.

        nnz : list of int
            A list holding the number of non-zero values of the next example.

        y : list of float
            The target value of the next example.

        sample_weight : list of float
            The weight of the next example.

        Returns
        -------
        current_index : int
            Index of current sample.
        """
        current_index = self._get_random_index()
        self._sample(x_data, x_ind, nnz, y, sample_weight, current_index)
        return current_index

    def shuffle(self, seed):
        """Permutes the ordering of examples."""
        # Fisher-Yates shuffle
        ind = self.index
        n = self.n_samples
        for i in range(n - 1):
            j = i + our_rand_r(seed) % (n - i)
            ind[i], ind[j] = ind[j], ind[i]

    def _get_next_index(self):
        current_index = self.current_index
        if current_index >= (self.n_samples - 1):
            current_index = -1

        current_index += 1
        self.current_index = current_index
        return self.current_index

    def _get_random_index(self):
        n = self.n_samples
        current_index = our_rand_r(self.seed) % n
        self.current_index = current_index
        return current_index

    def _sample(self, x_data, x_ind, nnz, y, sample_weight, current_index):
        pass

    def _shuffle_py(self, seed):
        """python function used for easy testing"""
        self.shuffle(seed)

    def _next_py(self):
        """python function used for easy testing"""
        current_index = self._get_next_index()
        return self._sample_py(current_index)

    def _random_py(self):
        """python function used for easy testing"""
        current_index = self._get_random_index()
        return self._sample_py(current_index)

    def _sample_py(self, current_index):
        """python function used for easy testing"""
        x_data = [0]
        x_indices = [0]
        nnz = [0]
        y = [0]
        sample_weight = [0]

        # call _sample in Python
        self._sample(x_data, x_indices, nnz, y, sample_weight, current_index)

        # transform the pointed data in numpy CSR array
        x_data = np.empty(nnz[0], dtype=np.float64)
        x_indices = np.empty(nnz[0], dtype=np.int32)
        x_indptr = np.asarray([0, nnz[0]], dtype=np.int32)

        for j in range(nnz[0]):
            x_data[j] = x_data[0][j]
            x_indices[j] = x_indices[0][j]

        sample_idx = self.index[current_index]

        return (
            (np.asarray(x_data), np.asarray(x_indices), np.asarray(x_indptr)),
            y[0],
            sample_weight[0],
            sample_idx,
        )



# Assigning aliases
ArrayDataset64 = SequentialDataset64
CSRDataset64 = SequentialDataset64
SequentialDataset32 = SequentialDataset64
ArrayDataset32 = SequentialDataset64
CSRDataset32 = SequentialDataset64
