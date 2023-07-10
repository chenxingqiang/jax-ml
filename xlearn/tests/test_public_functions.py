from importlib import import_module
from inspect import signature
from numbers import Integral, Real

import pytest

from xlearn.utils._param_validation import (
    Interval,
    InvalidParameterError,
    generate_invalid_param_val,
    generate_valid_param,
    make_constraint,
)


def _get_func_info(func_module):
    module_name, func_name = func_module.rsplit(".", 1)
    module = import_module(module_name)
    func = getattr(module, func_name)

    func_sig = signature(func)
    func_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    # The parameters `*args` and `**kwargs` are ignored since we cannot generate
    # constraints.
    required_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    return func, func_name, func_params, required_params


def _check_function_param_validation(
    func, func_name, func_params, required_params, parameter_constraints
):
    """Check that an informative error is raised when the value of a parameter does not
    have an appropriate type or value.
    """
    # generate valid values for the required parameters
    valid_required_params = {}
    for param_name in required_params:
        if parameter_constraints[param_name] == "no_validation":
            valid_required_params[param_name] = 1
        else:
            valid_required_params[param_name] = generate_valid_param(
                make_constraint(parameter_constraints[param_name][0])
            )

    # check that there is a constraint for each parameter
    if func_params:
        validation_params = parameter_constraints.keys()
        unexpected_params = set(validation_params) - set(func_params)
        missing_params = set(func_params) - set(validation_params)
        err_msg = (
            "Mismatch between _parameter_constraints and the parameters of"
            f" {func_name}.\nConsider the unexpected parameters {unexpected_params} and"
            f" expected but missing parameters {missing_params}\n"
        )
        assert set(validation_params) == set(func_params), err_msg

    # this object does not have a valid type for sure for all params
    param_with_bad_type = type("BadType", (), {})()

    for param_name in func_params:
        constraints = parameter_constraints[param_name]

        if constraints == "no_validation":
            # This parameter is not validated
            continue

        # Mixing an interval of reals and an interval of integers must be avoided.
        if any(
            isinstance(constraint, Interval) and constraint.type == Integral
            for constraint in constraints
        ) and any(
            isinstance(constraint, Interval) and constraint.type == Real
            for constraint in constraints
        ):
            raise ValueError(
                f"The constraint for parameter {param_name} of {func_name} can't have a"
                " mix of intervals of Integral and Real types. Use the type"
                " RealNotInt instead of Real."
            )

        match = (
            rf"The '{param_name}' parameter of {func_name} must be .* Got .* instead."
        )

        err_msg = (
            f"{func_name} does not raise an informative error message when the "
            f"parameter {param_name} does not have a valid type. If any Python type "
            "is valid, the constraint should be 'no_validation'."
        )

        # First, check that the error is raised if param doesn't match any valid type.
        with pytest.raises(InvalidParameterError, match=match):
            func(**{**valid_required_params, param_name: param_with_bad_type})
            pytest.fail(err_msg)

        # Then, for constraints that are more than a type constraint, check that the
        # error is raised if param does match a valid type but does not match any valid
        # value for this type.
        constraints = [make_constraint(constraint)
                       for constraint in constraints]

        for constraint in constraints:
            try:
                bad_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                continue

            err_msg = (
                f"{func_name} does not raise an informative error message when the "
                f"parameter {param_name} does not have a valid value.\n"
                "Constraints should be disjoint. For instance "
                "[StrOptions({'a_string'}), str] is not a acceptable set of "
                "constraint because generating an invalid string for the first "
                "constraint will always produce a valid string for the second "
                "constraint."
            )

            with pytest.raises(InvalidParameterError, match=match):
                func(**{**valid_required_params, param_name: bad_value})
                pytest.fail(err_msg)


PARAM_VALIDATION_FUNCTION_LIST = [
    "xlearn.calibration.calibration_curve",
    "xlearn.cluster.cluster_optics_dbscan",
    "xlearn.cluster.compute_optics_graph",
    "xlearn.cluster.estimate_bandwidth",
    "xlearn.cluster.kmeans_plusplus",
    "xlearn.cluster.cluster_optics_xi",
    "xlearn.cluster.ward_tree",
    "xlearn.covariance.empirical_covariance",
    "xlearn.covariance.ledoit_wolf_shrinkage",
    "xlearn.covariance.log_likelihood",
    "xlearn.covariance.shrunk_covariance",
    "xlearn.datasets.clear_data_home",
    "xlearn.datasets.dump_svmlight_file",
    "xlearn.datasets.fetch_20newsgroups",
    "xlearn.datasets.fetch_20newsgroups_vectorized",
    "xlearn.datasets.fetch_california_housing",
    "xlearn.datasets.fetch_covtype",
    "xlearn.datasets.fetch_kddcup99",
    "xlearn.datasets.fetch_lfw_pairs",
    "xlearn.datasets.fetch_lfw_people",
    "xlearn.datasets.fetch_olivetti_faces",
    "xlearn.datasets.fetch_rcv1",
    "xlearn.datasets.fetch_openml",
    "xlearn.datasets.fetch_species_distributions",
    "xlearn.datasets.get_data_home",
    "xlearn.datasets.load_breast_cancer",
    "xlearn.datasets.load_diabetes",
    "xlearn.datasets.load_digits",
    "xlearn.datasets.load_files",
    "xlearn.datasets.load_iris",
    "xlearn.datasets.load_linnerud",
    "xlearn.datasets.load_sample_image",
    "xlearn.datasets.load_svmlight_file",
    "xlearn.datasets.load_svmlight_files",
    "xlearn.datasets.load_wine",
    "xlearn.datasets.make_biclusters",
    "xlearn.datasets.make_blobs",
    "xlearn.datasets.make_checkerboard",
    "xlearn.datasets.make_circles",
    "xlearn.datasets.make_classification",
    "xlearn.datasets.make_friedman1",
    "xlearn.datasets.make_friedman2",
    "xlearn.datasets.make_friedman3",
    "xlearn.datasets.make_gaussian_quantiles",
    "xlearn.datasets.make_hastie_10_2",
    "xlearn.datasets.make_low_rank_matrix",
    "xlearn.datasets.make_moons",
    "xlearn.datasets.make_multilabel_classification",
    "xlearn.datasets.make_regression",
    "xlearn.datasets.make_s_curve",
    "xlearn.datasets.make_sparse_coded_signal",
    "xlearn.datasets.make_sparse_spd_matrix",
    "xlearn.datasets.make_sparse_uncorrelated",
    "xlearn.datasets.make_spd_matrix",
    "xlearn.datasets.make_swiss_roll",
    "xlearn.decomposition.sparse_encode",
    "xlearn.feature_extraction.grid_to_graph",
    "xlearn.feature_extraction.img_to_graph",
    "xlearn.feature_extraction.image.extract_patches_2d",
    "xlearn.feature_extraction.image.reconstruct_from_patches_2d",
    "xlearn.feature_selection.chi2",
    "xlearn.feature_selection.f_classif",
    "xlearn.feature_selection.f_regression",
    "xlearn.feature_selection.mutual_info_classif",
    "xlearn.feature_selection.mutual_info_regression",
    "xlearn.feature_selection.r_regression",
    "xlearn.inspection.partial_dependence",
    "xlearn.inspection.permutation_importance",
    "xlearn.isotonic.check_increasing",
    "xlearn.isotonic.isotonic_regression",
    "xlearn.linear_model.enet_path",
    "xlearn.linear_model.lars_path",
    "xlearn.linear_model.lars_path_gram",
    "xlearn.linear_model.lasso_path",
    "xlearn.linear_model.orthogonal_mp",
    "xlearn.linear_model.orthogonal_mp_gram",
    "xlearn.linear_model.ridge_regression",
    "xlearn.manifold.trustworthiness",
    "xlearn.metrics.accuracy_score",
    "xlearn.manifold.smacof",
    "xlearn.metrics.auc",
    "xlearn.metrics.average_precision_score",
    "xlearn.metrics.balanced_accuracy_score",
    "xlearn.metrics.brier_score_loss",
    "xlearn.metrics.calinski_harabasz_score",
    "xlearn.metrics.check_scoring",
    "xlearn.metrics.completeness_score",
    "xlearn.metrics.class_likelihood_ratios",
    "xlearn.metrics.classification_report",
    "xlearn.metrics.cluster.adjusted_mutual_info_score",
    "xlearn.metrics.cluster.contingency_matrix",
    "xlearn.metrics.cluster.entropy",
    "xlearn.metrics.cluster.fowlkes_mallows_score",
    "xlearn.metrics.cluster.homogeneity_completeness_v_measure",
    "xlearn.metrics.cluster.normalized_mutual_info_score",
    "xlearn.metrics.cluster.silhouette_samples",
    "xlearn.metrics.cluster.silhouette_score",
    "xlearn.metrics.cohen_kappa_score",
    "xlearn.metrics.confusion_matrix",
    "xlearn.metrics.consensus_score",
    "xlearn.metrics.coverage_error",
    "xlearn.metrics.d2_absolute_error_score",
    "xlearn.metrics.d2_pinball_score",
    "xlearn.metrics.d2_tweedie_score",
    "xlearn.metrics.davies_bouldin_score",
    "xlearn.metrics.dcg_score",
    "xlearn.metrics.det_curve",
    "xlearn.metrics.explained_variance_score",
    "xlearn.metrics.f1_score",
    "xlearn.metrics.fbeta_score",
    "xlearn.metrics.get_scorer",
    "xlearn.metrics.hamming_loss",
    "xlearn.metrics.hinge_loss",
    "xlearn.metrics.homogeneity_score",
    "xlearn.metrics.jaccard_score",
    "xlearn.metrics.label_ranking_average_precision_score",
    "xlearn.metrics.label_ranking_loss",
    "xlearn.metrics.log_loss",
    "xlearn.metrics.make_scorer",
    "xlearn.metrics.matthews_corrcoef",
    "xlearn.metrics.max_error",
    "xlearn.metrics.mean_absolute_error",
    "xlearn.metrics.mean_absolute_percentage_error",
    "xlearn.metrics.mean_gamma_deviance",
    "xlearn.metrics.mean_pinball_loss",
    "xlearn.metrics.mean_poisson_deviance",
    "xlearn.metrics.mean_squared_error",
    "xlearn.metrics.mean_squared_log_error",
    "xlearn.metrics.mean_tweedie_deviance",
    "xlearn.metrics.median_absolute_error",
    "xlearn.metrics.multilabel_confusion_matrix",
    "xlearn.metrics.mutual_info_score",
    "xlearn.metrics.ndcg_score",
    "xlearn.metrics.pair_confusion_matrix",
    "xlearn.metrics.adjusted_rand_score",
    "xlearn.metrics.pairwise.additive_chi2_kernel",
    "xlearn.metrics.pairwise.chi2_kernel",
    "xlearn.metrics.pairwise.cosine_distances",
    "xlearn.metrics.pairwise.cosine_similarity",
    "xlearn.metrics.pairwise.euclidean_distances",
    "xlearn.metrics.pairwise.haversine_distances",
    "xlearn.metrics.pairwise.laplacian_kernel",
    "xlearn.metrics.pairwise.linear_kernel",
    "xlearn.metrics.pairwise.manhattan_distances",
    "xlearn.metrics.pairwise.nan_euclidean_distances",
    "xlearn.metrics.pairwise.paired_cosine_distances",
    "xlearn.metrics.pairwise.paired_distances",
    "xlearn.metrics.pairwise.paired_euclidean_distances",
    "xlearn.metrics.pairwise.paired_manhattan_distances",
    "xlearn.metrics.pairwise.pairwise_distances_argmin_min",
    "xlearn.metrics.pairwise.pairwise_kernels",
    "xlearn.metrics.pairwise.polynomial_kernel",
    "xlearn.metrics.pairwise.rbf_kernel",
    "xlearn.metrics.pairwise.sigmoid_kernel",
    "xlearn.metrics.pairwise_distances",
    "xlearn.metrics.pairwise_distances_argmin",
    "xlearn.metrics.pairwise_distances_chunked",
    "xlearn.metrics.precision_recall_curve",
    "xlearn.metrics.precision_recall_fscore_support",
    "xlearn.metrics.precision_score",
    "xlearn.metrics.r2_score",
    "xlearn.metrics.rand_score",
    "xlearn.metrics.recall_score",
    "xlearn.metrics.roc_auc_score",
    "xlearn.metrics.roc_curve",
    "xlearn.metrics.top_k_accuracy_score",
    "xlearn.metrics.v_measure_score",
    "xlearn.metrics.zero_one_loss",
    "xlearn.model_selection.cross_val_predict",
    "xlearn.model_selection.cross_val_score",
    "xlearn.model_selection.cross_validate",
    "xlearn.model_selection.learning_curve",
    "xlearn.model_selection.permutation_test_score",
    "xlearn.model_selection.train_test_split",
    "xlearn.model_selection.validation_curve",
    "xlearn.neighbors.sort_graph_by_row_values",
    "xlearn.preprocessing.add_dummy_feature",
    "xlearn.preprocessing.binarize",
    "xlearn.preprocessing.label_binarize",
    "xlearn.preprocessing.normalize",
    "xlearn.preprocessing.scale",
    "xlearn.random_projection.johnson_lindenstrauss_min_dim",
    "xlearn.svm.l1_min_c",
    "xlearn.tree.export_graphviz",
    "xlearn.tree.export_text",
    "xlearn.tree.plot_tree",
    "xlearn.utils.gen_batches",
    "xlearn.utils.gen_even_slices",
    "xlearn.utils.resample",
    "xlearn.utils.safe_mask",
    "xlearn.utils.extmath.randomized_svd",
    "xlearn.utils.class_weight.compute_class_weight",
    "xlearn.utils.class_weight.compute_sample_weight",
    "xlearn.utils.graph.single_source_shortest_path_length",
]


@pytest.mark.parametrize("func_module", PARAM_VALIDATION_FUNCTION_LIST)
def test_function_param_validation(func_module):
    """Check param validation for public functions that are not wrappers around
    estimators.
    """
    func, func_name, func_params, required_params = _get_func_info(func_module)

    parameter_constraints = getattr(func, "_skl_parameter_constraints")

    _check_function_param_validation(
        func, func_name, func_params, required_params, parameter_constraints
    )


PARAM_VALIDATION_CLASS_WRAPPER_LIST = [
    ("xlearn.cluster.affinity_propagation", "xlearn.cluster.AffinityPropagation"),
    ("xlearn.cluster.k_means", "xlearn.cluster.KMeans"),
    ("xlearn.cluster.mean_shift", "xlearn.cluster.MeanShift"),
    ("xlearn.cluster.spectral_clustering", "xlearn.cluster.SpectralClustering"),
    ("xlearn.covariance.graphical_lasso", "xlearn.covariance.GraphicalLasso"),
    ("xlearn.covariance.ledoit_wolf", "xlearn.covariance.LedoitWolf"),
    ("xlearn.covariance.oas", "xlearn.covariance.OAS"),
    ("xlearn.decomposition.dict_learning",
     "xlearn.decomposition.DictionaryLearning"),
    ("xlearn.decomposition.fastica", "xlearn.decomposition.FastICA"),
    ("xlearn.decomposition.non_negative_factorization", "xlearn.decomposition.NMF"),
    ("xlearn.preprocessing.maxabs_scale", "xlearn.preprocessing.MaxAbsScaler"),
    ("xlearn.preprocessing.minmax_scale", "xlearn.preprocessing.MinMaxScaler"),
    ("xlearn.preprocessing.power_transform",
     "xlearn.preprocessing.PowerTransformer"),
    (
        "xlearn.preprocessing.quantile_transform",
        "xlearn.preprocessing.QuantileTransformer",
    ),
    ("xlearn.preprocessing.robust_scale", "xlearn.preprocessing.RobustScaler"),
]


@pytest.mark.parametrize(
    "func_module, class_module", PARAM_VALIDATION_CLASS_WRAPPER_LIST
)
def test_class_wrapper_param_validation(func_module, class_module):
    """Check param validation for public functions that are wrappers around
    estimators.
    """
    func, func_name, func_params, required_params = _get_func_info(func_module)

    module_name, class_name = class_module.rsplit(".", 1)
    module = import_module(module_name)
    klass = getattr(module, class_name)

    parameter_constraints_func = getattr(func, "_skl_parameter_constraints")
    parameter_constraints_class = getattr(klass, "_parameter_constraints")
    parameter_constraints = {
        **parameter_constraints_class,
        **parameter_constraints_func,
    }
    parameter_constraints = {
        k: v for k, v in parameter_constraints.items() if k in func_params
    }

    _check_function_param_validation(
        func, func_name, func_params, required_params, parameter_constraints
    )
