.. _api_ref:

=============
API Reference
=============

This is the class and function reference of jax-learn. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.
For reference on concepts repeated across the API, see :ref:`glossary`.


:mod:`xlearn.base`: Base classes and utility functions
=======================================================

.. automodule:: xlearn.base
    :no-members:
    :no-inherited-members:

Base classes
------------
.. currentmodule:: xlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   base.BaseEstimator
   base.BiclusterMixin
   base.ClassifierMixin
   base.ClusterMixin
   base.DensityMixin
   base.RegressorMixin
   base.TransformerMixin
   base.MetaEstimatorMixin
   base.OneToOneFeatureMixin
   base.ClassNamePrefixFeaturesOutMixin
   feature_selection.SelectorMixin

Functions
---------
.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   base.clone
   base.is_classifier
   base.is_regressor
   config_context
   get_config
   set_config
   show_versions

.. _calibration_ref:

:mod:`xlearn.calibration`: Probability Calibration
===================================================

.. automodule:: xlearn.calibration
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`calibration` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   calibration.CalibratedClassifierCV


.. autosummary::
   :toctree: generated/
   :template: function.rst

   calibration.calibration_curve

.. _cluster_ref:

:mod:`xlearn.cluster`: Clustering
==================================

.. automodule:: xlearn.cluster
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`clustering` and :ref:`biclustering` sections for
further details.

Classes
-------
.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   cluster.AffinityPropagation
   cluster.AgglomerativeClustering
   cluster.Birch
   cluster.DBSCAN
   cluster.HDBSCAN
   cluster.FeatureAgglomeration
   cluster.KMeans
   cluster.BisectingKMeans
   cluster.MiniBatchKMeans
   cluster.MeanShift
   cluster.OPTICS
   cluster.SpectralClustering
   cluster.SpectralBiclustering
   cluster.SpectralCoclustering

Functions
---------
.. autosummary::
   :toctree: generated/
   :template: function.rst

   cluster.affinity_propagation
   cluster.cluster_optics_dbscan
   cluster.cluster_optics_xi
   cluster.compute_optics_graph
   cluster.dbscan
   cluster.estimate_bandwidth
   cluster.k_means
   cluster.kmeans_plusplus
   cluster.mean_shift
   cluster.spectral_clustering
   cluster.ward_tree

.. _compose_ref:

:mod:`xlearn.compose`: Composite Estimators
============================================

.. automodule:: xlearn.compose
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`combining_estimators` section for further
details.

.. currentmodule:: xlearn

.. autosummary::
    :toctree: generated
    :template: class.rst

    compose.ColumnTransformer
    compose.TransformedTargetRegressor

.. autosummary::
   :toctree: generated/
   :template: function.rst

   compose.make_column_transformer
   compose.make_column_selector

.. _covariance_ref:

:mod:`xlearn.covariance`: Covariance Estimators
================================================

.. automodule:: xlearn.covariance
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`covariance` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   covariance.EmpiricalCovariance
   covariance.EllipticEnvelope
   covariance.GraphicalLasso
   covariance.GraphicalLassoCV
   covariance.LedoitWolf
   covariance.MinCovDet
   covariance.OAS
   covariance.ShrunkCovariance

.. autosummary::
   :toctree: generated/
   :template: function.rst

   covariance.empirical_covariance
   covariance.graphical_lasso
   covariance.ledoit_wolf
   covariance.ledoit_wolf_shrinkage
   covariance.oas
   covariance.shrunk_covariance

.. _cross_decomposition_ref:

:mod:`xlearn.cross_decomposition`: Cross decomposition
=======================================================

.. automodule:: xlearn.cross_decomposition
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`cross_decomposition` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   cross_decomposition.CCA
   cross_decomposition.PLSCanonical
   cross_decomposition.PLSRegression
   cross_decomposition.PLSSVD

.. _datasets_ref:

:mod:`xlearn.datasets`: Datasets
=================================

.. automodule:: xlearn.datasets
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`datasets` section for further details.

Loaders
-------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.clear_data_home
   datasets.dump_svmlight_file
   datasets.fetch_20newsgroups
   datasets.fetch_20newsgroups_vectorized
   datasets.fetch_california_housing
   datasets.fetch_covtype
   datasets.fetch_kddcup99
   datasets.fetch_lfw_pairs
   datasets.fetch_lfw_people
   datasets.fetch_olivetti_faces
   datasets.fetch_openml
   datasets.fetch_rcv1
   datasets.fetch_species_distributions
   datasets.get_data_home
   datasets.load_breast_cancer
   datasets.load_diabetes
   datasets.load_digits
   datasets.load_files
   datasets.load_iris
   datasets.load_linnerud
   datasets.load_sample_image
   datasets.load_sample_images
   datasets.load_svmlight_file
   datasets.load_svmlight_files
   datasets.load_wine

Samples generator
-----------------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.make_biclusters
   datasets.make_blobs
   datasets.make_checkerboard
   datasets.make_circles
   datasets.make_classification
   datasets.make_friedman1
   datasets.make_friedman2
   datasets.make_friedman3
   datasets.make_gaussian_quantiles
   datasets.make_hastie_10_2
   datasets.make_low_rank_matrix
   datasets.make_moons
   datasets.make_multilabel_classification
   datasets.make_regression
   datasets.make_s_curve
   datasets.make_sparse_coded_signal
   datasets.make_sparse_spd_matrix
   datasets.make_sparse_uncorrelated
   datasets.make_spd_matrix
   datasets.make_swiss_roll


.. _decomposition_ref:

:mod:`xlearn.decomposition`: Matrix Decomposition
==================================================

.. automodule:: xlearn.decomposition
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`decompositions` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   decomposition.DictionaryLearning
   decomposition.FactorAnalysis
   decomposition.FastICA
   decomposition.IncrementalPCA
   decomposition.KernelPCA
   decomposition.LatentDirichletAllocation
   decomposition.MiniBatchDictionaryLearning
   decomposition.MiniBatchSparsePCA
   decomposition.NMF
   decomposition.MiniBatchNMF
   decomposition.PCA
   decomposition.SparsePCA
   decomposition.SparseCoder
   decomposition.TruncatedSVD

.. autosummary::
   :toctree: generated/
   :template: function.rst

   decomposition.dict_learning
   decomposition.dict_learning_online
   decomposition.fastica
   decomposition.non_negative_factorization
   decomposition.sparse_encode

.. _lda_ref:

:mod:`xlearn.discriminant_analysis`: Discriminant Analysis
===========================================================

.. automodule:: xlearn.discriminant_analysis
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`lda_qda` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated
   :template: class.rst

   discriminant_analysis.LinearDiscriminantAnalysis
   discriminant_analysis.QuadraticDiscriminantAnalysis

.. _dummy_ref:

:mod:`xlearn.dummy`: Dummy estimators
======================================

.. automodule:: xlearn.dummy
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`model_evaluation` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   dummy.DummyClassifier
   dummy.DummyRegressor

.. autosummary::
   :toctree: generated/
   :template: function.rst

.. _ensemble_ref:

:mod:`xlearn.ensemble`: Ensemble Methods
=========================================

.. automodule:: xlearn.ensemble
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`ensemble` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ensemble.AdaBoostClassifier
   ensemble.AdaBoostRegressor
   ensemble.BaggingClassifier
   ensemble.BaggingRegressor
   ensemble.ExtraTreesClassifier
   ensemble.ExtraTreesRegressor
   ensemble.GradientBoostingClassifier
   ensemble.GradientBoostingRegressor
   ensemble.IsolationForest
   ensemble.RandomForestClassifier
   ensemble.RandomForestRegressor
   ensemble.RandomTreesEmbedding
   ensemble.StackingClassifier
   ensemble.StackingRegressor
   ensemble.VotingClassifier
   ensemble.VotingRegressor
   ensemble.HistGradientBoostingRegressor
   ensemble.HistGradientBoostingClassifier


.. autosummary::
   :toctree: generated/
   :template: function.rst


.. _exceptions_ref:

:mod:`xlearn.exceptions`: Exceptions and warnings
==================================================

.. automodule:: xlearn.exceptions
   :no-members:
   :no-inherited-members:

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   exceptions.ConvergenceWarning
   exceptions.DataConversionWarning
   exceptions.DataDimensionalityWarning
   exceptions.EfficiencyWarning
   exceptions.FitFailedWarning
   exceptions.InconsistentVersionWarning
   exceptions.NotFittedError
   exceptions.UndefinedMetricWarning


:mod:`xlearn.experimental`: Experimental
=========================================

.. automodule:: xlearn.experimental
   :no-members:
   :no-inherited-members:

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/

   experimental.enable_iterative_imputer
   experimental.enable_halving_search_cv


.. _feature_extraction_ref:

:mod:`xlearn.feature_extraction`: Feature Extraction
=====================================================

.. automodule:: xlearn.feature_extraction
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`feature_extraction` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_extraction.DictVectorizer
   feature_extraction.FeatureHasher

From images
-----------

.. automodule:: xlearn.feature_extraction.image
   :no-members:
   :no-inherited-members:

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   feature_extraction.image.extract_patches_2d
   feature_extraction.image.grid_to_graph
   feature_extraction.image.img_to_graph
   feature_extraction.image.reconstruct_from_patches_2d

   :template: class.rst

   feature_extraction.image.PatchExtractor

.. _text_feature_extraction_ref:

From text
---------

.. automodule:: xlearn.feature_extraction.text
   :no-members:
   :no-inherited-members:

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_extraction.text.CountVectorizer
   feature_extraction.text.HashingVectorizer
   feature_extraction.text.TfidfTransformer
   feature_extraction.text.TfidfVectorizer


.. _feature_selection_ref:

:mod:`xlearn.feature_selection`: Feature Selection
===================================================

.. automodule:: xlearn.feature_selection
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`feature_selection` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_selection.GenericUnivariateSelect
   feature_selection.SelectPercentile
   feature_selection.SelectKBest
   feature_selection.SelectFpr
   feature_selection.SelectFdr
   feature_selection.SelectFromModel
   feature_selection.SelectFwe
   feature_selection.SequentialFeatureSelector
   feature_selection.RFE
   feature_selection.RFECV
   feature_selection.VarianceThreshold

.. autosummary::
   :toctree: generated/
   :template: function.rst

   feature_selection.chi2
   feature_selection.f_classif
   feature_selection.f_regression
   feature_selection.r_regression
   feature_selection.mutual_info_classif
   feature_selection.mutual_info_regression


.. _gaussian_process_ref:

:mod:`xlearn.gaussian_process`: Gaussian Processes
===================================================

.. automodule:: xlearn.gaussian_process
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`gaussian_process` section for further details.

.. currentmodule:: xlearn

.. autosummary::
  :toctree: generated/
  :template: class.rst

  gaussian_process.GaussianProcessClassifier
  gaussian_process.GaussianProcessRegressor

Kernels:

.. autosummary::
  :toctree: generated/
  :template: class_with_call.rst

  gaussian_process.kernels.CompoundKernel
  gaussian_process.kernels.ConstantKernel
  gaussian_process.kernels.DotProduct
  gaussian_process.kernels.ExpSineSquared
  gaussian_process.kernels.Exponentiation
  gaussian_process.kernels.Hyperparameter
  gaussian_process.kernels.Kernel
  gaussian_process.kernels.Matern
  gaussian_process.kernels.PairwiseKernel
  gaussian_process.kernels.Product
  gaussian_process.kernels.RBF
  gaussian_process.kernels.RationalQuadratic
  gaussian_process.kernels.Sum
  gaussian_process.kernels.WhiteKernel


.. _impute_ref:

:mod:`xlearn.impute`: Impute
=============================

.. automodule:: xlearn.impute
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`Impute` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   impute.SimpleImputer
   impute.IterativeImputer
   impute.MissingIndicator
   impute.KNNImputer


.. _inspection_ref:

:mod:`xlearn.inspection`: Inspection
=====================================

.. automodule:: xlearn.inspection
   :no-members:
   :no-inherited-members:

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   inspection.partial_dependence
   inspection.permutation_importance

Plotting
--------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: display_only_from_estimator.rst

   inspection.DecisionBoundaryDisplay
   inspection.PartialDependenceDisplay

.. _isotonic_ref:

:mod:`xlearn.isotonic`: Isotonic regression
============================================

.. automodule:: xlearn.isotonic
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`isotonic` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   isotonic.IsotonicRegression

.. autosummary::
   :toctree: generated
   :template: function.rst

   isotonic.check_increasing
   isotonic.isotonic_regression


.. _kernel_approximation_ref:

:mod:`xlearn.kernel_approximation`: Kernel Approximation
=========================================================

.. automodule:: xlearn.kernel_approximation
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`kernel_approximation` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   kernel_approximation.AdditiveChi2Sampler
   kernel_approximation.Nystroem
   kernel_approximation.PolynomialCountSketch
   kernel_approximation.RBFSampler
   kernel_approximation.SkewedChi2Sampler

.. _kernel_ridge_ref:

:mod:`xlearn.kernel_ridge`: Kernel Ridge Regression
====================================================

.. automodule:: xlearn.kernel_ridge
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`kernel_ridge` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   kernel_ridge.KernelRidge

.. _linear_model_ref:

:mod:`xlearn.linear_model`: Linear Models
==========================================

.. automodule:: xlearn.linear_model
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`linear_model` section for further details.

The following subsections are only rough guidelines: the same estimator can
fall into multiple categories, depending on its parameters.

.. currentmodule:: xlearn

Linear classifiers
------------------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.LogisticRegression
   linear_model.LogisticRegressionCV
   linear_model.PassiveAggressiveClassifier
   linear_model.Perceptron
   linear_model.RidgeClassifier
   linear_model.RidgeClassifierCV
   linear_model.SGDClassifier
   linear_model.SGDOneClassSVM

Classical linear regressors
---------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.LinearRegression
   linear_model.Ridge
   linear_model.RidgeCV
   linear_model.SGDRegressor

Regressors with variable selection
----------------------------------

The following estimators have built-in variable selection fitting
procedures, but any estimator using a L1 or elastic-net penalty also
performs variable selection: typically :class:`~linear_model.SGDRegressor`
or :class:`~xlearn.linear_model.SGDClassifier` with an appropriate penalty.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.ElasticNet
   linear_model.ElasticNetCV
   linear_model.Lars
   linear_model.LarsCV
   linear_model.Lasso
   linear_model.LassoCV
   linear_model.LassoLars
   linear_model.LassoLarsCV
   linear_model.LassoLarsIC
   linear_model.OrthogonalMatchingPursuit
   linear_model.OrthogonalMatchingPursuitCV

Bayesian regressors
-------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.ARDRegression
   linear_model.BayesianRidge

Multi-task linear regressors with variable selection
----------------------------------------------------

These estimators fit multiple regression problems (or tasks) jointly, while
inducing sparse coefficients. While the inferred coefficients may differ
between the tasks, they are constrained to agree on the features that are
selected (non-zero coefficients).

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.MultiTaskElasticNet
   linear_model.MultiTaskElasticNetCV
   linear_model.MultiTaskLasso
   linear_model.MultiTaskLassoCV

Outlier-robust regressors
-------------------------

Any estimator using the Huber loss would also be robust to outliers, e.g.
:class:`~linear_model.SGDRegressor` with ``loss='huber'``.

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.HuberRegressor
   linear_model.QuantileRegressor
   linear_model.RANSACRegressor
   linear_model.TheilSenRegressor

Generalized linear models (GLM) for regression
----------------------------------------------

These models allow for response variables to have error distributions other
than a normal distribution:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.PoissonRegressor
   linear_model.TweedieRegressor
   linear_model.GammaRegressor


Miscellaneous
-------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   linear_model.PassiveAggressiveRegressor
   linear_model.enet_path
   linear_model.lars_path
   linear_model.lars_path_gram
   linear_model.lasso_path
   linear_model.orthogonal_mp
   linear_model.orthogonal_mp_gram
   linear_model.ridge_regression


.. _manifold_ref:

:mod:`xlearn.manifold`: Manifold Learning
==========================================

.. automodule:: xlearn.manifold
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`manifold` section for further details.

.. currentmodule:: xlearn

.. autosummary::
    :toctree: generated
    :template: class.rst

    manifold.Isomap
    manifold.LocallyLinearEmbedding
    manifold.MDS
    manifold.SpectralEmbedding
    manifold.TSNE

.. autosummary::
    :toctree: generated
    :template: function.rst

    manifold.locally_linear_embedding
    manifold.smacof
    manifold.spectral_embedding
    manifold.trustworthiness


.. _metrics_ref:

:mod:`xlearn.metrics`: Metrics
===============================

See the :ref:`model_evaluation` section and the :ref:`metrics` section of the
user guide for further details.

.. automodule:: xlearn.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: xlearn

Model Selection Interface
-------------------------
See the :ref:`scoring_parameter` section of the user guide for further
details.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.check_scoring
   metrics.get_scorer
   metrics.get_scorer_names
   metrics.make_scorer

Classification metrics
----------------------

See the :ref:`classification_metrics` section of the user guide for further
details.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.accuracy_score
   metrics.auc
   metrics.average_precision_score
   metrics.balanced_accuracy_score
   metrics.brier_score_loss
   metrics.class_likelihood_ratios
   metrics.classification_report
   metrics.cohen_kappa_score
   metrics.confusion_matrix
   metrics.dcg_score
   metrics.det_curve
   metrics.f1_score
   metrics.fbeta_score
   metrics.hamming_loss
   metrics.hinge_loss
   metrics.jaccard_score
   metrics.log_loss
   metrics.matthews_corrcoef
   metrics.multilabel_confusion_matrix
   metrics.ndcg_score
   metrics.precision_recall_curve
   metrics.precision_recall_fscore_support
   metrics.precision_score
   metrics.recall_score
   metrics.roc_auc_score
   metrics.roc_curve
   metrics.top_k_accuracy_score
   metrics.zero_one_loss

Regression metrics
------------------

See the :ref:`regression_metrics` section of the user guide for further
details.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.explained_variance_score
   metrics.max_error
   metrics.mean_absolute_error
   metrics.mean_squared_error
   metrics.mean_squared_log_error
   metrics.median_absolute_error
   metrics.mean_absolute_percentage_error
   metrics.r2_score
   metrics.mean_poisson_deviance
   metrics.mean_gamma_deviance
   metrics.mean_tweedie_deviance
   metrics.d2_tweedie_score
   metrics.mean_pinball_loss
   metrics.d2_pinball_score
   metrics.d2_absolute_error_score

Multilabel ranking metrics
--------------------------
See the :ref:`multilabel_ranking_metrics` section of the user guide for further
details.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.coverage_error
   metrics.label_ranking_average_precision_score
   metrics.label_ranking_loss


Clustering metrics
------------------

See the :ref:`clustering_evaluation` section of the user guide for further
details.

.. automodule:: xlearn.metrics.cluster
   :no-members:
   :no-inherited-members:

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.adjusted_mutual_info_score
   metrics.adjusted_rand_score
   metrics.calinski_harabasz_score
   metrics.davies_bouldin_score
   metrics.completeness_score
   metrics.cluster.contingency_matrix
   metrics.cluster.pair_confusion_matrix
   metrics.fowlkes_mallows_score
   metrics.homogeneity_completeness_v_measure
   metrics.homogeneity_score
   metrics.mutual_info_score
   metrics.normalized_mutual_info_score
   metrics.rand_score
   metrics.silhouette_score
   metrics.silhouette_samples
   metrics.v_measure_score

Biclustering metrics
--------------------

See the :ref:`biclustering_evaluation` section of the user guide for
further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.consensus_score

Distance metrics
----------------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   metrics.DistanceMetric

Pairwise metrics
----------------

See the :ref:`metrics` section of the user guide for further details.

.. automodule:: xlearn.metrics.pairwise
   :no-members:
   :no-inherited-members:

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.pairwise.additive_chi2_kernel
   metrics.pairwise.chi2_kernel
   metrics.pairwise.cosine_similarity
   metrics.pairwise.cosine_distances
   metrics.pairwise.distance_metrics
   metrics.pairwise.euclidean_distances
   metrics.pairwise.haversine_distances
   metrics.pairwise.kernel_metrics
   metrics.pairwise.laplacian_kernel
   metrics.pairwise.linear_kernel
   metrics.pairwise.manhattan_distances
   metrics.pairwise.nan_euclidean_distances
   metrics.pairwise.pairwise_kernels
   metrics.pairwise.polynomial_kernel
   metrics.pairwise.rbf_kernel
   metrics.pairwise.sigmoid_kernel
   metrics.pairwise.paired_euclidean_distances
   metrics.pairwise.paired_manhattan_distances
   metrics.pairwise.paired_cosine_distances
   metrics.pairwise.paired_distances
   metrics.pairwise_distances
   metrics.pairwise_distances_argmin
   metrics.pairwise_distances_argmin_min
   metrics.pairwise_distances_chunked


Plotting
--------

See the :ref:`visualizations` section of the user guide for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: display_all_class_methods.rst

   metrics.ConfusionMatrixDisplay
   metrics.DetCurveDisplay
   metrics.PrecisionRecallDisplay
   metrics.PredictionErrorDisplay
   metrics.RocCurveDisplay
   calibration.CalibrationDisplay

.. _mixture_ref:

:mod:`xlearn.mixture`: Gaussian Mixture Models
===============================================

.. automodule:: xlearn.mixture
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`mixture` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mixture.BayesianGaussianMixture
   mixture.GaussianMixture

.. _modelselection_ref:

:mod:`xlearn.model_selection`: Model Selection
===============================================

.. automodule:: xlearn.model_selection
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`cross_validation`, :ref:`grid_search` and
:ref:`learning_curve` sections for further details.

Splitter Classes
----------------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.GroupKFold
   model_selection.GroupShuffleSplit
   model_selection.KFold
   model_selection.LeaveOneGroupOut
   model_selection.LeavePGroupsOut
   model_selection.LeaveOneOut
   model_selection.LeavePOut
   model_selection.PredefinedSplit
   model_selection.RepeatedKFold
   model_selection.RepeatedStratifiedKFold
   model_selection.ShuffleSplit
   model_selection.StratifiedKFold
   model_selection.StratifiedShuffleSplit
   model_selection.StratifiedGroupKFold
   model_selection.TimeSeriesSplit

Splitter Functions
------------------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   model_selection.check_cv
   model_selection.train_test_split

.. _hyper_parameter_optimizers:

Hyper-parameter optimizers
--------------------------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.GridSearchCV
   model_selection.HalvingGridSearchCV
   model_selection.ParameterGrid
   model_selection.ParameterSampler
   model_selection.RandomizedSearchCV
   model_selection.HalvingRandomSearchCV


Model validation
----------------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   model_selection.cross_validate
   model_selection.cross_val_predict
   model_selection.cross_val_score
   model_selection.learning_curve
   model_selection.permutation_test_score
   model_selection.validation_curve

Visualization
-------------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: display_only_from_estimator.rst

   model_selection.LearningCurveDisplay
   model_selection.ValidationCurveDisplay

.. _multiclass_ref:

:mod:`xlearn.multiclass`: Multiclass classification
====================================================

.. automodule:: xlearn.multiclass
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`multiclass_classification` section for further details.

.. currentmodule:: xlearn

.. autosummary::
    :toctree: generated
    :template: class.rst

    multiclass.OneVsRestClassifier
    multiclass.OneVsOneClassifier
    multiclass.OutputCodeClassifier

.. _multioutput_ref:

:mod:`xlearn.multioutput`: Multioutput regression and classification
=====================================================================

.. automodule:: xlearn.multioutput
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`multilabel_classification`,
:ref:`multiclass_multioutput_classification`, and
:ref:`multioutput_regression` sections for further details.

.. currentmodule:: xlearn

.. autosummary::
    :toctree: generated
    :template: class.rst

    multioutput.ClassifierChain
    multioutput.MultiOutputRegressor
    multioutput.MultiOutputClassifier
    multioutput.RegressorChain

.. _naive_bayes_ref:

:mod:`xlearn.naive_bayes`: Naive Bayes
=======================================

.. automodule:: xlearn.naive_bayes
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`naive_bayes` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   naive_bayes.BernoulliNB
   naive_bayes.CategoricalNB
   naive_bayes.ComplementNB
   naive_bayes.GaussianNB
   naive_bayes.MultinomialNB


.. _neighbors_ref:

:mod:`xlearn.neighbors`: Nearest Neighbors
===========================================

.. automodule:: xlearn.neighbors
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`neighbors` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   neighbors.BallTree
   neighbors.KDTree
   neighbors.KernelDensity
   neighbors.KNeighborsClassifier
   neighbors.KNeighborsRegressor
   neighbors.KNeighborsTransformer
   neighbors.LocalOutlierFactor
   neighbors.RadiusNeighborsClassifier
   neighbors.RadiusNeighborsRegressor
   neighbors.RadiusNeighborsTransformer
   neighbors.NearestCentroid
   neighbors.NearestNeighbors
   neighbors.NeighborhoodComponentsAnalysis

.. autosummary::
   :toctree: generated/
   :template: function.rst

   neighbors.kneighbors_graph
   neighbors.radius_neighbors_graph
   neighbors.sort_graph_by_row_values

.. _neural_network_ref:

:mod:`xlearn.neural_network`: Neural network models
====================================================

.. automodule:: xlearn.neural_network
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`neural_networks_supervised` and :ref:`neural_networks_unsupervised` sections for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   neural_network.BernoulliRBM
   neural_network.MLPClassifier
   neural_network.MLPRegressor

.. _pipeline_ref:

:mod:`xlearn.pipeline`: Pipeline
=================================

.. automodule:: xlearn.pipeline
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`combining_estimators` section for further
details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pipeline.FeatureUnion
   pipeline.Pipeline

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pipeline.make_pipeline
   pipeline.make_union

.. _preprocessing_ref:

:mod:`xlearn.preprocessing`: Preprocessing and Normalization
=============================================================

.. automodule:: xlearn.preprocessing
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`preprocessing` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.Binarizer
   preprocessing.FunctionTransformer
   preprocessing.KBinsDiscretizer
   preprocessing.KernelCenterer
   preprocessing.LabelBinarizer
   preprocessing.LabelEncoder
   preprocessing.MultiLabelBinarizer
   preprocessing.MaxAbsScaler
   preprocessing.MinMaxScaler
   preprocessing.Normalizer
   preprocessing.OneHotEncoder
   preprocessing.OrdinalEncoder
   preprocessing.PolynomialFeatures
   preprocessing.PowerTransformer
   preprocessing.QuantileTransformer
   preprocessing.RobustScaler
   preprocessing.SplineTransformer
   preprocessing.StandardScaler
   preprocessing.TargetEncoder

.. autosummary::
   :toctree: generated/
   :template: function.rst

   preprocessing.add_dummy_feature
   preprocessing.binarize
   preprocessing.label_binarize
   preprocessing.maxabs_scale
   preprocessing.minmax_scale
   preprocessing.normalize
   preprocessing.quantile_transform
   preprocessing.robust_scale
   preprocessing.scale
   preprocessing.power_transform


.. _random_projection_ref:

:mod:`xlearn.random_projection`: Random projection
===================================================

.. automodule:: xlearn.random_projection
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`random_projection` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   random_projection.GaussianRandomProjection
   random_projection.SparseRandomProjection

.. autosummary::
   :toctree: generated/
   :template: function.rst

   random_projection.johnson_lindenstrauss_min_dim


.. _semi_supervised_ref:

:mod:`xlearn.semi_supervised`: Semi-Supervised Learning
========================================================

.. automodule:: xlearn.semi_supervised
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`semi_supervised` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   semi_supervised.LabelPropagation
   semi_supervised.LabelSpreading
   semi_supervised.SelfTrainingClassifier


.. _svm_ref:

:mod:`xlearn.svm`: Support Vector Machines
===========================================

.. automodule:: xlearn.svm
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`svm` section for further details.

Estimators
----------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   svm.LinearSVC
   svm.LinearSVR
   svm.NuSVC
   svm.NuSVR
   svm.OneClassSVM
   svm.SVC
   svm.SVR

.. autosummary::
   :toctree: generated/
   :template: function.rst

   svm.l1_min_c

.. _tree_ref:

:mod:`xlearn.tree`: Decision Trees
===================================

.. automodule:: xlearn.tree
   :no-members:
   :no-inherited-members:

**User guide:** See the :ref:`tree` section for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   tree.DecisionTreeClassifier
   tree.DecisionTreeRegressor
   tree.ExtraTreeClassifier
   tree.ExtraTreeRegressor

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tree.export_graphviz
   tree.export_text

Plotting
--------

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tree.plot_tree

.. _utils_ref:

:mod:`xlearn.utils`: Utilities
===============================

.. automodule:: xlearn.utils
   :no-members:
   :no-inherited-members:

**Developer guide:** See the :ref:`developers-utils` page for further details.

.. currentmodule:: xlearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   utils.Bunch

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.arrayfuncs.min_pos
   utils.as_float_array
   utils.assert_all_finite
   utils.check_X_y
   utils.check_array
   utils.check_scalar
   utils.check_consistent_length
   utils.check_random_state
   utils.class_weight.compute_class_weight
   utils.class_weight.compute_sample_weight
   utils.deprecated
   utils.estimator_checks.check_estimator
   utils.estimator_checks.parametrize_with_checks
   utils.estimator_html_repr
   utils.extmath.safe_sparse_dot
   utils.extmath.randomized_range_finder
   utils.extmath.randomized_svd
   utils.extmath.fast_logdet
   utils.extmath.density
   utils.extmath.weighted_mode
   utils.gen_batches
   utils.gen_even_slices
   utils.graph.single_source_shortest_path_length
   utils.indexable
   utils.metaestimators.available_if
   utils.multiclass.type_of_target
   utils.multiclass.is_multilabel
   utils.multiclass.unique_labels
   utils.murmurhash3_32
   utils.resample
   utils._safe_indexing
   utils.safe_mask
   utils.safe_sqr
   utils.shuffle
   utils.sparsefuncs.incr_mean_variance_axis
   utils.sparsefuncs.inplace_column_scale
   utils.sparsefuncs.inplace_row_scale
   utils.sparsefuncs.inplace_swap_row
   utils.sparsefuncs.inplace_swap_column
   utils.sparsefuncs.mean_variance_axis
   utils.sparsefuncs.inplace_csr_column_scale
   utils.sparsefuncs_fast.inplace_csr_row_normalize_l1
   utils.sparsefuncs_fast.inplace_csr_row_normalize_l2
   utils.random.sample_without_replacement
   utils.validation.check_is_fitted
   utils.validation.check_memory
   utils.validation.check_symmetric
   utils.validation.column_or_1d
   utils.validation.has_fit_parameter
   utils.metadata_routing.get_routing_for_object
   utils.metadata_routing.MetadataRouter
   utils.metadata_routing.MetadataRequest
   utils.metadata_routing.MethodMapping
   utils.metadata_routing.process_routing

Specific utilities to list jax-learn components:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.discovery.all_estimators
   utils.discovery.all_displays
   utils.discovery.all_functions

Utilities from joblib:

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.parallel.delayed
   utils.parallel_backend
   utils.register_parallel_backend

.. autosummary::
   :toctree: generated/
   :template: class.rst

   utils.parallel.Parallel


Recently deprecated
===================
