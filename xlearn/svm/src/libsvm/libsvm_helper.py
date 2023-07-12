import numpy as np
from typing import List, Tuple
from svm import svm_node, svm_model, svm_parameter, svm_predict, svm_predict_values, svm_predict_probability, svm_set_print_string_function

def dense_to_libsvm(x: np.ndarray) -> List[svm_node]:
    nrow, ncol = x.shape
    node = []
    for i in range(nrow):
        values = x[i, :].tolist()
        ind = i + 1
        node.append(svm_node(ind, values))
    return node

def set_parameter(param: svm_parameter, svm_type: int, kernel_type: int, degree: int,
                  gamma: float, coef0: float, nu: float, cache_size: float, C: float,
                  eps: float, p: float, shrinking: int, probability: int, nr_weight: int,
                  weight_label: List[int], weight: List[float], max_iter: int, random_seed: int):
    param.svm_type = svm_type
    param.kernel_type = kernel_type
    param.degree = degree
    param.coef0 = coef0
    param.nu = nu
    param.cache_size = cache_size
    param.C = C
    param.eps = eps
    param.p = p
    param.shrinking = shrinking
    param.probability = probability
    param.nr_weight = nr_weight
    param.weight_label = weight_label
    param.weight = weight
    param.gamma = gamma
    param.max_iter = max_iter
    param.random_seed = random_seed

def set_problem(problem: svm_problem, X: np.ndarray, Y: np.ndarray, sample_weight: np.ndarray, kernel_type: int):
    problem.l = X.shape[0]
    problem.y = Y.tolist()
    problem.x = dense_to_libsvm(X)
    problem.W = sample_weight.tolist()

def set_model(param: svm_parameter, nr_class: int, SV: np.ndarray, support: np.ndarray,
              sv_coef: np.ndarray, rho: np.ndarray, nSV: np.ndarray, probA: np.ndarray,
              probB: np.ndarray) -> svm_model:
    model = svm_model()
    model.nr_class = nr_class
    model.param = param
    model.l = support.shape[0]
    model.SV = dense_to_libsvm(SV)

    if param.svm_type < 2:
        model.nSV = nSV.tolist()
        model.label = list(range(nr_class))

    model.sv_coef = sv_coef.tolist()
    model.rho = rho.tolist()
    model.probA = probA.tolist()
    model.probB = probB.tolist()

    model.free_sv = 0
    return model

def get_l(model: svm_model) -> int:
    return model.l

def get_nr(model: svm_model) -> int:
    return model.nr_class

def copy_predict(predict: np.ndarray, model: svm_model) -> np.ndarray:
    n = predict.shape[0]
    dec_values = np.zeros(n)
    for i in range(n):
        dec_values[i] = svm_predict(model, predict[i])
    return dec_values

def copy_predict_values(predict: np.ndarray, model: svm_model, nr_class: int) -> np.ndarray:
    n = predict.shape[0]
    dec_values = np.zeros((n, nr_class))
    for i in range(n):
        svm_predict_values(model, predict[i], dec_values[i])
    return dec_values

def copy_predict_proba(predict: np.ndarray, model: svm_model, nr_class: int) -> np.ndarray:
    n = predict.shape[0]
    proba = np.zeros((n, nr_class))
    for i in range(n):
        svm_predict_probability(model, predict[i], proba[i])
    return proba

def free_model(model: svm_model):
    del model

def free_param(param: svm_parameter):
    del param

def set_verbosity(verbosity_flag: bool):
    if verbosity_flag:
        svm_set_print_string_function(print)
    else:
        svm_set_print_string_function(lambda s: None)
