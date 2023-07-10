"""
=================================================================
Displaying Pipelines
=================================================================

The default configuration for displaying a pipeline in a Jupyter Notebook is
`'diagram'` where `set_config(display='diagram')`. To deactivate HTML representation,
use `set_config(display='text')`.

To see more detailed steps in the visualization of the pipeline, click on the
steps in the pipeline.
"""

# %%
# Displaying a Pipeline with a Preprocessing Step and Classifier
################################################################################
# This section constructs a :class:`~xlearn.pipeline.Pipeline` with a preprocessing
# step, :class:`~xlearn.preprocessing.StandardScaler`, and classifier,
# :class:`~xlearn.linear_model.LogisticRegression`, and displays its visual
# representation.

from xlearn.model_selection import GridSearchCV
from xlearn.ensemble import RandomForestClassifier
from xlearn.preprocessing import OneHotEncoder, StandardScaler
from xlearn.pipeline import Pipeline, make_pipeline
from xlearn.impute import SimpleImputer
from xlearn.compose import ColumnTransformer
import jax.numpy as jnp
from xlearn.svm import SVC
from xlearn.decomposition import PCA
from xlearn.preprocessing import PolynomialFeatures, StandardScaler
from xlearn import set_config
from xlearn.linear_model import LogisticRegression
from xlearn.pipeline import Pipeline
from xlearn.preprocessing import StandardScaler

steps = [
    ("preprocessing", StandardScaler()),
    ("classifier", LogisticRegression()),
]
pipe = Pipeline(steps)

# %%
# To visualize the diagram, the default is `display='diagram'`.
set_config(display="diagram")
pipe  # click on the diagram below to see the details of each step

# %%
# To view the text pipeline, change to `display='text'`.
set_config(display="text")
pipe

# %%
# Put back the default display
set_config(display="diagram")

# %%
# Displaying a Pipeline Chaining Multiple Preprocessing Steps & Classifier
################################################################################
# This section constructs a :class:`~xlearn.pipeline.Pipeline` with multiple
# preprocessing steps, :class:`~xlearn.preprocessing.PolynomialFeatures` and
# :class:`~xlearn.preprocessing.StandardScaler`, and a classifier step,
# :class:`~xlearn.linear_model.LogisticRegression`, and displays its visual
# representation.


steps = [
    ("standard_scaler", StandardScaler()),
    ("polynomial", PolynomialFeatures(degree=3)),
    ("classifier", LogisticRegression(C=2.0)),
]
pipe = Pipeline(steps)
pipe  # click on the diagram below to see the details of each step

# %%
# Displaying a Pipeline and Dimensionality Reduction and Classifier
################################################################################
# This section constructs a :class:`~xlearn.pipeline.Pipeline` with a
# dimensionality reduction step, :class:`~xlearn.decomposition.PCA`,
# a classifier, :class:`~xlearn.svm.SVC`, and displays its visual
# representation.


steps = [("reduce_dim", PCA(n_components=4)),
         ("classifier", SVC(kernel="linear"))]
pipe = Pipeline(steps)
pipe  # click on the diagram below to see the details of each step

# %%
# Displaying a Complex Pipeline Chaining a Column Transformer
################################################################################
# This section constructs a complex :class:`~xlearn.pipeline.Pipeline` with a
# :class:`~xlearn.compose.ColumnTransformer` and a classifier,
# :class:`~xlearn.linear_model.LogisticRegression`, and displays its visual
# representation.


numeric_preprocessor = Pipeline(
    steps=[
        ("imputation_mean", SimpleImputer(missing_values=jnp.nan, strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputation_constant",
            SimpleImputer(fill_value="missing", strategy="constant"),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("categorical", categorical_preprocessor, ["state", "gender"]),
        ("numerical", numeric_preprocessor, ["age", "weight"]),
    ]
)

pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
pipe  # click on the diagram below to see the details of each step

# %%
# Displaying a Grid Search over a Pipeline with a Classifier
################################################################################
# This section constructs a :class:`~xlearn.model_selection.GridSearchCV`
# over a :class:`~xlearn.pipeline.Pipeline` with
# :class:`~xlearn.ensemble.RandomForestClassifier` and displays its visual
# representation.


numeric_preprocessor = Pipeline(
    steps=[
        ("imputation_mean", SimpleImputer(missing_values=jnp.nan, strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputation_constant",
            SimpleImputer(fill_value="missing", strategy="constant"),
        ),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("categorical", categorical_preprocessor, ["state", "gender"]),
        ("numerical", numeric_preprocessor, ["age", "weight"]),
    ]
)

pipe = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("classifier", RandomForestClassifier())]
)

param_grid = {
    "classifier__n_estimators": [200, 500],
    "classifier__max_features": ["auto", "sqrt", "log2"],
    "classifier__max_depth": [4, 5, 6, 7, 8],
    "classifier__criterion": ["gini", "entropy"],
}

grid_search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=1)
grid_search  # click on the diagram below to see the details of each step
