"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<xlearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

"""

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

# %%
# Generate data
# -------------
#
# In order to learn good latent representations from a small dataset, we
# artificially generate more labeled data by perturbing the training data with
# linear shifts of 1 pixel in each direction.

import matplotlib.pyplot as plt
from xlearn import metrics
from xlearn.base import clone
from xlearn.pipeline import Pipeline
from xlearn.neural_network import BernoulliRBM
from xlearn import linear_model
import jax.numpy as jnp
from scipy.ndimage import convolve

from xlearn import datasets
from xlearn.model_selection import train_test_split
from xlearn.preprocessing import minmax_scale


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    ]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode="constant", weights=w).ravel()

    X = jnp.concatenate(
        [X] + [jnp.apply_along_axis(shift, 1, X, vector)
               for vector in direction_vectors]
    )
    Y = jnp.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


X, y = datasets.load_digits(return_X_y=True)
X = jnp.asarray(X, "float32")
X, Y = nudge_dataset(X, y)
X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# %%
# Models definition
# -----------------
#
# We build a classification pipeline with a BernoulliRBM feature extractor and
# a :class:`LogisticRegression <xlearn.linear_model.LogisticRegression>`
# classifier.


logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(
    steps=[("rbm", rbm), ("logistic", logistic)])

# %%
# Training
# --------
#
# The hyperparameters of the entire model (learning rate, hidden layer size,
# regularization) were optimized by grid search, but the search is not
# reproduced here because of runtime constraints.


# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 10

# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000

# Training RBM-Logistic Pipeline
rbm_features_classifier.fit(X_train, Y_train)

# Training the Logistic regression classifier directly on the pixel
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.0
raw_pixel_classifier.fit(X_train, Y_train)

# %%
# Evaluation
# ----------


Y_pred = rbm_features_classifier.predict(X_test)
print(
    "Logistic regression using RBM features:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)

# %%
Y_pred = raw_pixel_classifier.predict(X_test)
print(
    "Logistic regression using raw pixel features:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)

# %%
# The features extracted by the BernoulliRBM help improve the classification
# accuracy with respect to the logistic regression on raw pixels.

# %%
# Plotting
# --------


plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
               interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
plt.suptitle("100 components extracted by RBM", fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
