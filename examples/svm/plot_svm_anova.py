"""
=================================================
SVM-Anova: SVM with univariate feature selection
=================================================

This example shows how to perform univariate feature selection before running a
SVC (support vector classifier) to improve the classification scores. We use
the iris dataset (4 features) and add 36 non-informative features. We can find
that our model achieves best performance when we select around 10% of features.

"""

# %%
# Load some data to play with
# ---------------------------
from xlearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xlearn.svm import SVC
from xlearn.preprocessing import StandardScaler
from xlearn.pipeline import Pipeline
from xlearn.feature_selection import SelectPercentile, f_classif
import jax.numpy as jnp

from xlearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Add non-informative features
rng = np.random.RandomState(0)
X = jnp.hstack((X, 2 * rng.random((X.shape[0], 36))))

# %%
# Create the pipeline
# -------------------

# Create a feature-selection transform, a scaler and an instance of SVM that we
# combine together to have a full-blown estimator

clf = Pipeline(
    [
        ("anova", SelectPercentile(f_classif)),
        ("scaler", StandardScaler()),
        ("svc", SVC(gamma="auto")),
    ]
)

# %%
# Plot the cross-validation score as a function of percentile of features
# -----------------------------------------------------------------------


score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)

for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    this_scores = cross_val_score(clf, X, y)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

plt.errorbar(percentiles, score_means, jnp.array(score_stds))
plt.title("Performance of the SVM-Anova varying the percentile of features selected")
plt.xticks(jnp.linspace(0, 100, 11, endpoint=True))
plt.xlabel("Percentile")
plt.ylabel("Accuracy Score")
plt.axis("tight")
plt.show()
