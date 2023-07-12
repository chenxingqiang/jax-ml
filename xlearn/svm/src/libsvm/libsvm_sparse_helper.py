import jax.numpy as jnp
from xlearn.svm.src.liblinear.liblinear_helper import print_null, print_string_stdout

from xlearn.svm.src.libsvm.svm_h import *


def csr_to_libsvm(values, indices, indptr, n_samples):
    sparse = []
    k = 0
    for i in range(n_samples):
        n = indptr[i+1] - indptr[i]
        temp = jnp.zeros(n+1, dtype=[('value', jnp.float64), ('index', jnp.int32)])
        temp['value'] = values[k:k+n]
        temp['index'] = indices[k:k+n] + 1
        k += n
        temp[n]['index'] = -1
        sparse.append(temp)
    return sparse


def set_parameter(svm_type, kernel_type, degree, gamma, coef0, nu, cache_size, C, eps, p, shrinking, probability, nr_weight, weight_label, weight, max_iter, random_seed):
    param = {
        'svm_type': svm_type,
        'kernel_type': kernel_type,
        'degree': degree,
        'gamma': gamma,
        'coef0': coef0,
        'nu': nu,
        'cache_size': cache_size,
        'C': C,
        'eps': eps,
        'p': p,
        'shrinking': shrinking,
        'probability': probability,
        'nr_weight': nr_weight,
        'weight_label': weight_label,
        'weight': weight,
        'max_iter': max_iter,
        'random_seed': random_seed
    }
    return param


def csr_set_problem(values, n_indices, indices, n_indptr, indptr, Y, sample_weight, kernel_type):
    problem = {
        'l': n_indptr[0] - 1,
        'y': Y,
        'x': csr_to_libsvm(values, indices, indptr, n_indptr[0] - 1),
        'W': sample_weight
    }
    return problem


def csr_set_model(param, nr_class, SV_data, SV_indices_dims, SV_indices, SV_indptr_dims, SV_intptr, sv_coef, rho, nSV, probA, probB):
    model = {
        'nr_class': nr_class,
        'SV': csr_to_libsvm(SV_data, SV_indices, SV_intptr, SV_indptr_dims[0] - 1),
        'sv_coef': sv_coef,
        'rho': -rho,
        'nSV': nSV,
        'probA': probA,
        'probB': probB,
        'param': param
    }
    return model


def csr_copy_SV(data, n_indices, indices, n_indptr, indptr, model, n_features):
    k = 0
    for i in range(model['l']):
        j = 0
        index = model['SV'][i][j]['index']
        while index >= 0:
            indices[k] = index - 1
            data[k] = model['SV'][i][j]['value']
            index = model['SV'][i][j+1]['index']
            j += 1
            k += 1
        n_indptr[i+1] = k
    return 0


def get_nonzero_SV(model):
    count = 0
    for i in range(model['l']):
        j = 0
        while model['SV'][i][j]['index'] != -1:
            j += 1
            count += 1
    return count


def csr_copy_predict(data_size, data, index_size, index, intptr_size, intptr, model, dec_values, blas_functions):
    t = dec_values
    predict_nodes = csr_to_libsvm(data, index, intptr, intptr_size[0] - 1)
    for i in range(intptr_size[0] - 1):
        t[0] = svm_csr_predict(model, predict_nodes[i], blas_functions)
        del predict_nodes[i]
        t += 1
    del predict_nodes
    return 0


def csr_copy_predict_values(data_size, data, index_size, index, intptr_size, intptr, model, dec_values, nr_class, blas_functions):
    predict_nodes = csr_to_libsvm(data, index, intptr, intptr_size[0] - 1)
    for i in range(intptr_size[0] - 1):
        svm_csr_predict_values(model, predict_nodes[i], dec_values[i*nr_class:(i+1)*nr_class], blas_functions)
        del predict_nodes[i]
    del predict_nodes
    return 0


def csr_copy_predict_proba(data_size, data, index_size, index, intptr_size, intptr, model, dec_values, blas_functions):
    predict_nodes = csr_to_libsvm(data, index, intptr, intptr_size[0] - 1)
    for i in range(intptr_size[0] - 1):
        svm_csr_predict_probability(model, predict_nodes[i], dec_values[i*model['nr_class']:(i+1)*model['nr_class']], blas_functions)
        del predict_nodes[i]
    del predict_nodes
    return 0


def get_nr(model):
    return model['nr_class']


def copy_intercept(data, model, dims):
    intercept = -model['rho']
    data[:] = jnp.where(intercept != 0, intercept, 0)


def copy_support(data, model):
    data[:] = model['sv_ind']


def copy_sv_coef(data, model):
    k = 0
    for i in range(model['nr_class'] - 1):
        data[k:k+model['l']] = model['sv_coef'][i]
        k += model['l']


def copy_n_iter(data, model):
    n_models = max(1, model['nr_class'] * (model['nr_class'] - 1) // 2)
    data[:] = model['n_iter'][:n_models]


def get_l(model):
    return model['l']


def copy_nSV(data, model):
    if model['label'] is not None:
        data[:] = model['nSV']


def copy_label(data, model):
    if model['label'] is not None:
        data[:] = model['label']


def copy_probA(data, model, dims):
    data[:] = model['probA']


def copy_probB(data, model, dims):
    data[:] = model['probB']


def free_problem(problem):
    for x in problem['x']:
        del x
    del problem['x']
    del problem
    return 0


def free_model(model):
    for i in range(model['l'] - 1, -1, -1):
        del model['SV'][i]
    del model['SV']
    del model['sv_coef']
    del model['rho']
    del model['label']
    del model['probA']
    del model['probB']
    del model['nSV']
    del model
    return 0


def free_param(param):
    del param
    return 0


def free_model_SV(model):
    for i in range(model['l'] - 1, -1, -1):
        del model['SV'][i]
    for i in range(model['nr_class'] - 1):
        del model['sv_coef'][i]
    return 0


def set_verbosity(verbosity_flag):
    if verbosity_flag:
        svm_set_print_string_function(print_string_stdout)
    else:
        svm_set_print_string_function(print_null)