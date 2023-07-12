import jax
import jax.numpy as jnp

import numpy as np

def info(fmt, *args):
    print(fmt % args)

def min(x, y):
    return x if x < y else y

def max(x, y):
    return x if x > y else y

def swap(x, y):
    return y, x

def clone(src, n):
    return jnp.array(src, copy=True)

def powi(base, times):
    tmp = base
    ret = 1.0

    while times > 0:
        if times % 2 == 1:
            ret *= tmp
        tmp = tmp * tmp
        times //= 2

    return ret

INF = jnp.inf
TAU = 1e-12

def print_string_stdout(s):
    print(s)

svm_print_string = print_string_stdout

def info(fmt, *args):
    buf = fmt % args
    svm_print_string(buf)

class Cache:
    def __init__(self, l, size):
        self.l = l
        self.size = size
        self.head = jnp.zeros(l, dtype=np.object)
        self.size //= jnp.dtype(np.float32).itemsize
        self.size -= l * jnp.dtype(np.object).itemsize // jnp.dtype(np.float32).itemsize
        self.size = max(self.size, 2 * l)
        self.lru_head = self.head[0]
        self.lru_head['prev'] = self.lru_head['next'] = self.lru_head

    def lru_delete(self, h):
        h['prev']['next'] = h['next']
        h['next']['prev'] = h['prev']

    def lru_insert(self, h):
        h['next'] = self.lru_head
        h['prev'] = self.lru_head['prev']
        h['prev']['next'] = h
        h['next']['prev'] = h

    def get_data(self, index, len):
        h = self.head[index]
        if h['len']:
            self.lru_delete(h)
        more = len - h['len']

        if more > 0:
            while self.size < more:
                old = self.lru_head['next']
                self.lru_delete(old)
                self.size += old['len']
                old['data'] = None
                old['len'] = 0

            h['data'] = jnp.zeros(len, dtype=np.float32)
            self.size -= more
            h['len'], len = len, h['len']

        self.lru_insert(h)
        return h['data'], len

    def swap_index(self, i, j):
        if i == j:
            return

        h_i = self.head[i]
        h_j = self.head[j]
        if h_i['len']:
            self.lru_delete(h_i)
        if h_j['len']:
            self.lru_delete(h_j)
        h_i['data'], h_j['data'] = swap(h_i['data'], h_j['data'])
        h_i['len'], h_j['len'] = h_j['len'], h_i['len']
        if h_i['len']:
            self.lru_insert(h_i)
        if h_j['len']:
            self.lru_insert(h_j)

        if i > j:
            i, j = j, i
        for h in self.lru_head['next']:
            if h['len'] > i:
                if h['len'] > j:
                    h['data'][i], h['data'][j] = swap(h['data'][i], h['data'][j])
                else:
                    self.lru_delete(h)
                    self.size += h['len']
                    h['data'] = None
                    h['len'] = 0

class QMatrix:
    def get_Q(self, column, len):
        pass

    def get_QD(self):
        pass

    def swap_index(self, i, j):
        pass

class Kernel(QMatrix):
    def __init__(self, l, x, param, blas_functions):
        self.l = l
        self.x = x
        self.x_square = None
        self.kernel_type = param.kernel_type
        self.degree = param.degree
        self.gamma = param.gamma
        self.coef0 = param.coef0
        self.m_blas = blas_functions

        if self.kernel_type == "linear":
            self.kernel_function = self.kernel_linear
        elif self.kernel_type == "poly":
            self.kernel_function = self.kernel_poly
        elif self.kernel_type == "rbf":
            self.kernel_function = self.kernel_rbf
        elif self.kernel_type == "sigmoid":
            self.kernel_function = self.kernel_sigmoid
        elif self.kernel_type == "precomputed":
            self.kernel_function = self.kernel_precomputed

        if self.kernel_type == "rbf":
            self.x_square = jnp.zeros(l)
            for i in range(l):
                self.x_square[i] = self.dot(x[i], x[i])

    def get_Q(self, column, len):
        Q = jnp.zeros(len, dtype=np.float32)
        QD = self.get_QD()
        for i in range(len):
            Q[i] = self.kernel_function(column, i)
        return Q

    def get_QD(self):
        if self.x_square is not None:
            return self.x_square
        else:
            return jnp.zeros(self.l)

    def swap_index(self, i, j):
        self.x[i], self.x[j] = swap(self.x[i], self.x[j])
        if self.x_square is not None:
            self.x_square[i], self.x_square[j] = swap(self.x_square[i], self.x_square[j])

    def dot(self, px, py):
        sum = 0
        while px['index'] != -1 and py['index'] != -1:
            if px['index'] == py['index']:
                sum += px['value'] * py['value']
                px += 1
                py += 1
            else:
                if px['index'] > py['index']:
                    py += 1
                else:
                    px += 1
        return sum

    def kernel_linear(self, i, j):
        return self.dot(self.x[i], self.x[j])

    def kernel_poly(self, i, j):
        return powi(self.gamma * self.dot(self.x[i], self.x[j]) + self.coef0, self.degree)

    def kernel_rbf(self, i, j):
        sum = 0
        for k in range(min(self.x[i]['dim'], self.x[j]['dim'])):
            sum += (self.x[i]['values'][k] - self.x[j]['values'][k]) ** 2
        return jnp.exp(-self.gamma * sum)

    def kernel_sigmoid(self, i, j):
        return jnp.tanh(self.gamma * self.dot(self.x[i], self.x[j]) + self.coef0)

    def kernel_precomputed(self, i, j):
        return self.x[i]['values'][self.x[j]['ind']]



class Solver:
    def __init__(self):
        pass

    class SolutionInfo:
        def __init__(self):
            self.obj = 0.0
            self.rho = 0.0
            self.upper_bound = None
            self.r = 0.0
            self.solve_timed_out = False
            self.n_iter = 0

    def Solve(self, l, Q, p_, y_, alpha_, C_, eps, si, shrinking, max_iter):
        self.l = l
        self.Q = Q
        self.QD = Q.get_QD()
        self.p = p_
        self.y = y_
        self.alpha = alpha_
        self.C = C_
        self.eps = eps
        self.unshrink = False
        si.solve_timed_out = False

        # initialize alpha_status
        self.alpha_status = [0] * l
        for i in range(l):
            self.update_alpha_status(i)

        # initialize active set (for shrinking)
        self.active_set = list(range(l))
        self.active_size = l

        # initialize gradient
        self.G = [0.0] * l
        self.G_bar = [0.0] * l
        for i in range(l):
            self.G[i] = self.p[i]
            self.G_bar[i] = 0.0
        for i in range(l):
            if not self.is_lower_bound(i):
                Q_i = self.Q.get_Q(i, l)
                alpha_i = self.alpha[i]
                for j in range(l):
                    self.G[j] += alpha_i * Q_i[j]
                if self.is_upper_bound(i):
                    for j in range(l):
                        self.G_bar[j] += self.get_C(i) * Q_i[j]

        # optimization step
        iter = 0
        counter = min(l, 1000) + 1

        while True:
            if max_iter != -1 and iter >= max_iter:
                print("WARN: libsvm Solver reached max_iter")
                si.solve_timed_out = True
                break

            # show progress and do shrinking
            counter -= 1
            if counter == 0:
                counter = min(l, 1000)
                if shrinking:
                    self.do_shrinking()
                print(".")

            i, j = self.select_working_set()
            if i == -1:
                # reconstruct the whole gradient
                self.reconstruct_gradient()
                # reset active set size and check
                self.active_size = l
                print("*")
                i, j = self.select_working_set()
                if i == -1:
                    break
                else:
                    counter = 1  # do shrinking next iteration

            iter += 1

            # update alpha[i] and alpha[j], handle bounds carefully
            Q_i = self.Q.get_Q(i, self.active_size)
            Q_j = self.Q.get_Q(j, self.active_size)
            C_i = self.get_C(i)
            C_j = self.get_C(j)
            old_alpha_i = self.alpha[i]
            old_alpha_j = self.alpha[j]

            if self.y[i] != self.y[j]:
                quad_coef = self.QD[i] + self.QD[j] + 2 * Q_i[j]
                if quad_coef <= 0:
                    quad_coef = 1e-12
                delta = (-self.G[i] - self.G[j]) / quad_coef
                diff = self.alpha[i] - self.alpha[j]
                self.alpha[i] += delta
                self.alpha[j] += delta

                if diff > 0:
                    if self.alpha[j] < 0:
                        self.alpha[j] = 0
                        self.alpha[i] = diff
                else:
                    if self.alpha[i] < 0:
                        self.alpha[i] = 0
                        self.alpha[j] = -diff

                if diff > C_i - C_j:
                    if self.alpha[i] > C_i:
                        self.alpha[i] = C_i
                        self.alpha[j] = C_i - diff
                else:
                    if self.alpha[j] > C_j:
                        self.alpha[j] = C_j
                        self.alpha[i] = C_j + diff
            else:
                quad_coef = self.QD[i] + self.QD[j] - 2 * Q_i[j]
                if quad_coef <= 0:
                    quad_coef = 1e-12
                delta = (self.G[i] - self.G[j]) / quad_coef
                sum = self.alpha[i] + self.alpha[j]
                self.alpha[i] -= delta
                self.alpha[j] += delta

                if sum > C_i:
                    if self.alpha[i] > C_i:
                        self.alpha[i] = C_i
                        self.alpha[j] = sum - C_i
                else:
                    if self.alpha[j] < 0:
                        self.alpha[j] = 0
                        self.alpha[i] = sum

                if sum > C_j:
                    if self.alpha[j] > C_j:
                        self.alpha[j] = C_j
                        self.alpha[i] = sum - C_j
                else:
                    if self.alpha[i] < 0:
                        self.alpha[i] = 0
                        self.alpha[j] = sum

            # update G
            delta_alpha_i = self.alpha[i] - old_alpha_i
            delta_alpha_j = self.alpha[j] - old_alpha_j
            for k in range(self.active_size):
                self.G[k] += Q_i[k] * delta_alpha_i + Q_j[k] * delta_alpha_j

            # update alpha_status and G_bar
            ui = self.is_upper_bound(i)
            uj = self.is_upper_bound(j)
            self.update_alpha_status(i)
            self.update_alpha_status(j)
            if ui != self.is_upper_bound(i):
                Q_i = self.Q.get_Q(i, l)
                if ui:
                    for k in range(l):
                        self.G_bar[k] -= C_i * Q_i[k]
                else:
                    for k in range(l):
                        self.G_bar[k] += C_i * Q_i[k]

            if uj != self.is_upper_bound(j):
                Q_j = self.Q.get_Q(j, l)
                if uj:
                    for k in range(l):
                        self.G_bar[k] -= C_j * Q_j[k]
                else:
                    for k in range(l):
                        self.G_bar[k] += C_j * Q_j[k]

        # calculate rho
        si.rho = self.calculate_rho()

        # calculate objective value
        v = 0.0
        for i in range(l):
            v += self.alpha[i] * (self.G[i] + self.p[i])
        si.obj = v / 2

        # put back the solution
        for i in range(l):
            alpha_[self.active_set[i]] = self.alpha[i]

        # store number of iterations
        si.n_iter = iter

        print("\noptimization finished, #iter =", iter)

        # clean up
        self.alpha_status = None
        self.active_set = None
        self.G = None
        self.G_bar = None

    def update_alpha_status(self, i):
        if self.alpha[i] >= self.get_C(i):
            self.alpha_status[i] = 1  # UPPER_BOUND
        elif self.alpha[i] <= 0:
            self.alpha_status[i] = -1  # LOWER_BOUND
        else:
            self.alpha_status[i] = 0  # FREE

    def is_upper_bound(self, i):
        return self.alpha_status[i] == 1

    def is_lower_bound(self, i):
        return self.alpha_status[i] == -1

    def get_C(self, i):
        return self.C[i]

    def swap_index(self, i, j):
        self.Q.swap_index(i, j)
        self.y[i], self.y[j] = self.y[j], self.y[i]
        self.G[i], self.G[j] = self.G[j], self.G[i]
        self.alpha_status[i], self.alpha_status[j] = self.alpha_status[j], self.alpha_status[i]
        self.alpha[i], self.alpha[j] = self.alpha[j], self.alpha[i]
        self.p[i], self.p[j] = self.p[j], self.p[i]
        self.active_set[i], self.active_set[j] = self.active_set[j], self.active_set[i]
        self.G_bar[i], self.G_bar[j] = self.G_bar[j], self.G_bar[i]
        self.C[i], self.C[j] = self.C[j], self.C[i]

    def reconstruct_gradient(self):
        # reconstruct inactive elements of G from G_bar and free variables
        if self.active_size == self.l:
            return

        nr_free = 0
        for j in range(self.active_size, self.l):
            self.G[j] = self.G_bar[j] + self.p[j]

        for j in range(self.active_size):
            if self.alpha_status[j] == 0:
                nr_free += 1

        if 2 * nr_free < self.active_size:
            print("\nWarning: using -h 0 may be faster\n")

        if nr_free * self.l > 2 * self.active_size * (self.l - self.active_size):
            for i in range(self.active_size, self.l):
                Q_i = self.Q.get_Q(i, self.active_size)
                for j in range(self.active_size):
                    if self.alpha_status[j] == 0:
                        self.G[i] += self.alpha[j] * Q_i[j]
        else:
            for i in range(self.active_size):
                if self.alpha_status[i] == 0:
                    Q_i = self.Q.get_Q(i, self.l)
                    alpha_i = self.alpha[i]
                    for j in range(self.active_size, self.l):
                        self.G[j] += alpha_i * Q_i[j]

    def select_working_set(self):
        Gmax = -float('inf')
        Gmax2 = -float('inf')
        Gmax_idx = -1
        Gmin_idx = -1
        obj_diff_min = float('inf')

        for t in range(self.active_size):
            if self.y[t] == 1:
                if not self.is_upper_bound(t):
                    if -self.G[t] >= Gmax:
                        Gmax = -self.G[t]
                        Gmax_idx = t
            else:
                if not self.is_lower_bound(t):
                    if self.G[t] >= Gmax:
                        Gmax = self.G[t]
                        Gmax_idx = t

        i = Gmax_idx
        Q_i = None
        if i != -1:
            Q_i = self.Q.get_Q(i, self.active_size)

        for j in range(self.active_size):
            if self.y[j] == 1:
                if not self.is_lower_bound(j):
                    grad_diff = Gmax + self.G[j]
                    if self.G[j] >= Gmax2:
                        Gmax2 = self.G[j]
                    if grad_diff > 0:
                        obj_diff = 0.0
                        quad_coef = self.QD[i] + self.QD[j] - 2.0 * self.y[i] * Q_i[j]
                        if quad_coef > 0:
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / 1e-12

                        if obj_diff <= obj_diff_min:
                            Gmin_idx = j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper_bound(j):
                    grad_diff = Gmax - self.G[j]
                    if -self.G[j] >= Gmax2:
                        Gmax2 = -self.G[j]
                    if grad_diff > 0:
                        obj_diff = 0.0
                        quad_coef = self.QD[i] + self.QD[j] + 2.0 * self.y[i] * Q_i[j]
                        if quad_coef > 0:
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / 1e-12

                        if obj_diff <= obj_diff_min:
                            Gmin_idx = j
                            obj_diff_min = obj_diff

        if Gmax + Gmax2 < self.eps or Gmin_idx == -1:
            return -1, -1

        return Gmax_idx, Gmin_idx

    def be_shrunk(self, i, Gmax1, Gmax2):
        if self.is_upper_bound(i):
            if self.y[i] == 1:
                return -self.G[i] > Gmax1
            else:
                return -self.G[i] > Gmax2
        elif self.is_lower_bound(i):
            if self.y[i] == 1:
                return self.G[i] > Gmax2
            else:
                return self.G[i] > Gmax1
        else:
            return False

    def do_shrinking(self):
        Gmax1 = -float('inf')  # max { -y_i * grad(f)_i | i in I_up(\alpha) }
        Gmax2 = -float('inf')  # max { y_i * grad(f)_i | i in I_low(\alpha) }

        # find maximal violating pair first
        for i in range(self.active_size):
            if self.y[i] == 1:
                if not self.is_upper_bound(i):
                    if -self.G[i] >= Gmax1:
                        Gmax1 = -self.G[i]
            else:
                if not self.is_upper_bound(i):
                    if -self.G[i] >= Gmax2:
                        Gmax2 = -self.G[i]

        if not self.unshrink and Gmax1 + Gmax2 <= self.eps * 10:
            self.unshrink = True
            self.reconstruct_gradient()
            self.active_size = self.l
            print("*")

        i = 0
        while i < self.active_size:
            if self.be_shrunk(i, Gmax1, Gmax2):
                self.active_size -= 1
                while self.active_size > i:
                    if not self.be_shrunk(self.active_size, Gmax1, Gmax2):
                        self.swap_index(i, self.active_size)
                        break
                    self.active_size -= 1
            i += 1

    def calculate_rho(self):
        r = 0.0
        nr_free = 0
        ub = float('inf')
        lb = -float('inf')
        sum_free = 0.0
        for i in range(self.active_size):
            yG = self.y[i] * self.G[i]
            if self.is_upper_bound(i):
                if self.y[i] == -1:
                    ub = min(ub, yG)
                else:
                    lb = max(lb, yG)
            elif self.is_lower_bound(i):
                if self.y[i] == 1:
                    ub = min(ub, yG)
                else:
                    lb = max(lb, yG)
            else:
                nr_free += 1
                sum_free += yG

        if nr_free > 0:
            r = sum_free / nr_free
        else:
            r = (ub + lb) / 2

        return r


class Solver_NU(Solver):
    def __init__(self):
        super().__init__()

    def Solve(self, l, Q, p, y, alpha, C_, eps, si, shrinking, max_iter):
        self.si = si
        super().Solve(l, Q, p, y, alpha, C_, eps, si, shrinking, max_iter)

    def select_working_set(self, out_i, out_j):
        Gmaxp = -np.inf
        Gmaxp2 = -np.inf
        Gmaxp_idx = -1

        Gmaxn = -np.inf
        Gmaxn2 = -np.inf
        Gmaxn_idx = -1

        Gmin_idx = -1
        obj_diff_min = jnp.inf

        for t in range(self.active_size):
            if self.y[t] == 1:
                if not self.is_upper_bound(t):
                    if -self.G[t] >= Gmaxp:
                        Gmaxp = -self.G[t]
                        Gmaxp_idx = t
            else:
                if not self.is_lower_bound(t):
                    if self.G[t] >= Gmaxn:
                        Gmaxn = self.G[t]
                        Gmaxn_idx = t

        ip = Gmaxp_idx
        in_ = Gmaxn_idx
        Q_ip = None
        Q_in = None
        if ip != -1:
            Q_ip = Q.get_Q(ip, self.active_size)
        if in_ != -1:
            Q_in = Q.get_Q(in_, self.active_size)

        for j in range(self.active_size):
            if self.y[j] == 1:
                if not self.is_lower_bound(j):
                    grad_diff = Gmaxp + self.G[j]
                    if self.G[j] >= Gmaxp2:
                        Gmaxp2 = self.G[j]
                    if grad_diff > 0:
                        obj_diff = 0.0
                        quad_coef = self.QD[ip] + self.QD[j] - 2 * Q_ip[j]
                        if quad_coef > 0:
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx = j
                            obj_diff_min = obj_diff
            else:
                if not self.is_upper_bound(j):
                    grad_diff = Gmaxn - self.G[j]
                    if -self.G[j] >= Gmaxn2:
                        Gmaxn2 = -self.G[j]
                    if grad_diff > 0:
                        obj_diff = 0.0
                        quad_coef = self.QD[in_] + self.QD[j] - 2 * Q_in[j]
                        if quad_coef > 0:
                            obj_diff = -(grad_diff * grad_diff) / quad_coef
                        else:
                            obj_diff = -(grad_diff * grad_diff) / TAU

                        if obj_diff <= obj_diff_min:
                            Gmin_idx = j
                            obj_diff_min = obj_diff

        if max(Gmaxp + Gmaxp2, Gmaxn + Gmaxn2) < self.eps or Gmin_idx == -1:
            return 1

        if self.y[Gmin_idx] == 1:
            out_i = Gmaxp_idx
        else:
            out_i = Gmaxn_idx
        out_j = Gmin_idx

        return 0

    def be_shrunk(self, i, Gmax1, Gmax2, Gmax3, Gmax4):
        if self.is_upper_bound(i):
            if self.y[i] == 1:
                return -self.G[i] > Gmax1
            else:
                return -self.G[i] > Gmax4
        elif self.is_lower_bound(i):
            if self.y[i] == 1:
                return self.G[i] > Gmax2
            else:
                return self.G[i] > Gmax3
        else:
            return False

    def do_shrinking(self):
        Gmax1 = -np.inf  # max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
        Gmax2 = -np.inf  # max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
        Gmax3 = -np.inf  # max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
        Gmax4 = -np.inf  # max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

        # find maximal violating pair first
        for i in range(self.active_size):
            if not self.is_upper_bound(i):
                if self.y[i] == 1:
                    if -self.G[i] > Gmax1:
                        Gmax1 = -self.G[i]
                else:
                    if -self.G[i] > Gmax4:
                        Gmax4 = -self.G[i]
            if not self.is_lower_bound(i):
                if self.y[i] == 1:
                    if self.G[i] > Gmax2:
                        Gmax2 = self.G[i]
                else:
                    if self.G[i] > Gmax3:
                        Gmax3 = self.G[i]

        if self.unshrink == False and max(Gmax1 + Gmax2, Gmax3 + Gmax4) <= self.eps * 10:
            self.unshrink = True
            self.reconstruct_gradient()
            self.active_size = self.l

        i = 0
        while i < self.active_size:
            if self.be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4):
                self.active_size -= 1
                while self.active_size > i:
                    if not self.be_shrunk(self.active_size, Gmax1, Gmax2, Gmax3, Gmax4):
                        self.swap_index(i, self.active_size)
                        break
                    self.active_size -= 1
            else:
                i += 1

    def calculate_rho(self):
        nr_free1 = 0
        nr_free2 = 0
        ub1 = jnp.inf
        ub2 = jnp.inf
        lb1 = -np.inf
        lb2 = -np.inf
        sum_free1 = 0
        sum_free2 = 0

        for i in range(self.active_size):
            if self.y[i] == 1:
                if self.is_upper_bound(i):
                    lb1 = max(lb1, self.G[i])
                elif self.is_lower_bound(i):
                    ub1 = min(ub1, self.G[i])
                else:
                    nr_free1 += 1
                    sum_free1 += self.G[i]
            else:
                if self.is_upper_bound(i):
                    lb2 = max(lb2, self.G[i])
                elif self.is_lower_bound(i):
                    ub2 = min(ub2, self.G[i])
                else:
                    nr_free2 += 1
                    sum_free2 += self.G[i]

        r1 = 0.0
        r2 = 0.0
        if nr_free1 > 0:
            r1 = sum_free1 / nr_free1
        else:
            r1 = (ub1 + lb1) / 2

        if nr_free2 > 0:
            r2 = sum_free2 / nr_free2
        else:
            r2 = (ub2 + lb2) / 2

        self.si.r = (r1 + r2) / 2
        return (r1 - r2) / 2

class SVC_Q(Kernel):
    def __init__(self, prob, param, y_, blas_functions):
        super().__init__(prob.l, prob.x, param, blas_functions)
        self.y = jnp.array(y_, dtype=np.int8)
        self.cache = Cache(prob.l, int(param.cache_size * (1 << 20)))
        self.QD = jnp.zeros(prob.l)
        for i in range(prob.l):
            self.QD[i] = self.kernel_function(i, i)

    def get_Q(self, i, length):
        data = jnp.zeros(length)
        start, j = self.cache.get_data(i, data, length)
        if start < length:
            for j in range(start, length):
                data[j] = self.y[i] * self.y[j] * self.kernel_function(i, j)
        return data

    def get_QD(self):
        return self.QD

    def swap_index(self, i, j):
        self.cache.swap_index(i, j)
        super().swap_index(i, j)
        self.y[i], self.y[j] = self.y[j], self.y[i]
        self.QD[i], self.QD[j] = self.QD[j], self.QD[i]

    def __del__(self):
        del self.y
        del self.cache
        del self.QD



class ONE_CLASS_Q(Kernel):
    def __init__(self, prob, param, blas_functions):
        super().__init__(prob.l, prob.x, param, blas_functions)
        self.cache = Cache(prob.l, int(param.cache_size*(1<<20)))
        self.QD = jnp.zeros(prob.l)
        for i in range(prob.l):
            self.QD[i] = self.kernel_function(i, i)

    def get_Q(self, i, length):
        data = jnp.zeros(length)
        start = self.cache.get_data(i, data, length)
        if start < length:
            for j in range(start, length):
                data[j] = self.kernel_function(i, j)
        return data

    def get_QD(self):
        return self.QD

    def swap_index(self, i, j):
        self.cache.swap_index(i, j)
        super().swap_index(i, j)
        self.QD[i], self.QD[j] = self.QD[j], self.QD[i]

    def __del__(self):
        del self.cache
        del self.QD


class SVR_Q(Kernel):
    def __init__(self, prob, param, blas_functions):
        super().__init__(prob.l, prob.x, param, blas_functions)
        self.l = prob.l
        self.cache = Cache(self.l, int(param.cache_size * (1 << 20)))
        self.QD = jnp.zeros(2*self.l)
        self.sign = jnp.zeros(2*self.l, dtype=np.int8)
        self.index = jnp.zeros(2*self.l, dtype=np.int32)
        for k in range(self.l):
            self.sign[k] = 1
            self.sign[k+self.l] = -1
            self.index[k] = k
            self.index[k+self.l] = k
            self.QD[k] = self.kernel_function(k, k)
            self.QD[k+self.l] = self.QD[k]
        self.buffer = [np.zeros(2*self.l, dtype=np.float64), jnp.zeros(2*self.l, dtype=np.float64)]
        self.next_buffer = 0

    def swap_index(self, i, j):
        self.sign[i], self.sign[j] = self.sign[j], self.sign[i]
        self.index[i], self.index[j] = self.index[j], self.index[i]
        self.QD[i], self.QD[j] = self.QD[j], self.QD[i]

    def get_Q(self, i, length):
        data = jnp.zeros(self.l, dtype=np.float64)
        real_i = self.index[i]
        if self.cache.get_data(real_i, data, self.l) < self.l:
            for j in range(self.l):
                data[j] = self.kernel_function(real_i, j)

        buf = self.buffer[self.next_buffer]
        self.next_buffer = 1 - self.next_buffer
        si = self.sign[i]
        for j in range(length):
            buf[j] = si * self.sign[j] * data[self.index[j]]
        return buf

    def get_QD(self):
        return self.QD

    def __del__(self):
        del self.cache
        del self.sign
        del self.index
        del self.buffer[0]
        del self.buffer[1]
        del self.QD



import jax
import jax.numpy as jnp

def solve_c_svc(prob, param, Cp, Cn):
    l = prob.l
    minus_ones = jnp.ones(l) * -1
    y = jnp.zeros(l)
    C = jnp.zeros(l)

    for i in range(l):
        alpha[i] = 0
        minus_ones[i] = -1
        if prob.y[i] > 0:
            y[i] = 1
            C[i] = prob.W[i] * Cp
        else:
            y[i] = -1
            C[i] = prob.W[i] * Cn

    def objective_function(alpha):
        return 0.5 * jnp.dot(alpha, alpha)

    def equality_constraint(alpha):
        return jnp.dot(y, alpha)

    def inequality_constraint(alpha):
        return jnp.maximum(0, C - alpha)

    def lagrangian(alpha):
        return objective_function(alpha) - jnp.dot(alpha, inequality_constraint(alpha))

    def loss_fn(alpha):
        return lagrangian(alpha)

    def grad_fn(alpha):
        return jax.grad(loss_fn)(alpha)

    def hessian_fn(alpha):
        return jax.hessian(loss_fn)(alpha)

    alpha = jax.scipy.optimize.minimize(loss_fn, alpha, method='trust-constr', jac=grad_fn, hess=hessian_fn,
                                         constraints=[{'type': 'eq', 'fun': equality_constraint},
                                                      {'type': 'ineq', 'fun': inequality_constraint}],
                                         options={'verbose': 1, 'maxiter': param.max_iter})

    sum_alpha = jnp.sum(alpha)

    if Cp == Cn:
        print("nu =", sum_alpha / (Cp * prob.l))

    alpha *= y

    return alpha


import jax
import jax.numpy as jnp

def solve_nu_svc(prob, param, blas_functions):
    l = prob.l
    nu = param.nu

    y = jnp.where(prob.y > 0, 1, -1)
    C = prob.W

    nu_l = jnp.sum(nu * C)
    sum_pos = nu_l / 2
    sum_neg = nu_l / 2

    alpha = jnp.zeros(l)
    for i in range(l):
        if y[i] == 1:
            alpha[i] = jnp.minimum(C[i], sum_pos)
            sum_pos -= alpha[i]
        else:
            alpha[i] = jnp.minimum(C[i], sum_neg)
            sum_neg -= alpha[i]

    zeros = jnp.zeros(l)

    s = Solver_NU()
    s.Solve(l, SVC_Q(prob, param, y, blas_functions), zeros, y,
            alpha, C, param.eps, s.si, param.shrinking, param.max_iter)
    r = s.si.r

    print("C =", 1 / r)

    alpha *= y / r
    s.si.upper_bound /= r
    s.si.rho /= r
    s.si.obj /= (r * r)


def solve_one_class(prob, param, blas_functions):
    l = prob["l"]
    zeros = jnp.zeros(l)
    ones = jnp.ones(l)
    C = jnp.array(prob["W"])

    nu_l = jnp.sum(C) * param["nu"]

    alpha = jnp.zeros(l)

    i = 0
    while nu_l > 0:
        alpha[i] = jnp.minimum(C[i], nu_l)
        nu_l -= alpha[i]
        i += 1

    alpha = jax.ops.index_update(alpha, jax.ops.index[i:], 0)

    si = Solver.SolutionInfo()

    s = Solver()
    s.Solve(l, ONE_CLASS_Q(prob, param, blas_functions), zeros, ones, alpha, C, param["eps"], si, param["shrinking"], param["max_iter"])

    return alpha




def solve_epsilon_svr(prob, param, blas_functions):
    l = prob.l
    alpha2 = jnp.zeros(2*l)
    linear_term = jnp.zeros(2*l)
    y = jnp.zeros(2*l, dtype=np.int8)
    C = jnp.zeros(2*l)

    for i in range(l):
        alpha2[i] = 0
        linear_term[i] = param.p - prob.y[i]
        y[i] = 1
        C[i] = prob.W[i] * param.C

        alpha2[i+l] = 0
        linear_term[i+l] = param.p + prob.y[i]
        y[i+l] = -1
        C[i+l] = prob.W[i] * param.C

    svr_q = SVR_Q(prob, param, blas_functions)
    linear_term = linear_term.reshape((2*l, 1))
    y = y.reshape((2*l, 1))
    C = C.reshape((2*l, 1))

    alpha2 = alpha2.reshape((2*l, 1))
    si = Solver.SolutionInfo()

    si = s.Solve(2*l, svr_q, linear_term, y, alpha2, C, param.eps, si, param.shrinking, param.max_iter)

    alpha = alpha2[:l] - alpha2[l:]
    sum_alpha = jnp.sum(np.abs(alpha))

    return alpha, sum_alpha



from jax.scipy.optimize import minimize

def solve_nu_svr(prob, param, alpha):
    l = prob.l
    C = jnp.concatenate([prob.W * param.C, prob.W * param.C])
    alpha2 = jnp.concatenate([jnp.minimum((jnp.sum(C) * param.nu) / 2, C)] * 2)
    linear_term = jnp.concatenate([-prob.y, prob.y])
    y = jnp.concatenate([jnp.ones(l), -jnp.ones(l)])

    def objective(x):
        return jnp.dot(x, linear_term)

    def constraint(x):
        return jnp.dot(x, y)

    bounds = [(0, c) for c in C]

    result = minimize(objective, alpha2, constraints={'type': 'eq', 'fun': constraint},
                      bounds=bounds, method='SLSQP', options={'maxiter': param.max_iter})

    si = Solver.SolutionInfo()
    si.r = -result.fun

    alpha[:l] = result.x[:l] - result.x[l:]

    return si

from typing import List, Tuple

class DecisionFunction:
    def __init__(self, alpha: List[float], rho: float, n_iter: int):
        self.alpha = alpha
        self.rho = rho
        self.n_iter = n_iter

def svm_train_one(prob: Tuple[np.ndarray, jnp.ndarray], param: dict, Cp: float, Cn: float, blas_functions):
    alpha = jnp.zeros(prob[0].shape[0])
    si = SolverSolutionInfo()
    if param['svm_type'] == 'C_SVC':
        si.upper_bound = jnp.zeros(prob[0].shape[0])
        solve_c_svc(prob, param, alpha, si, Cp, Cn, blas_functions)
    elif param['svm_type'] == 'NU_SVC':
        si.upper_bound = jnp.zeros(prob[0].shape[0])
        solve_nu_svc(prob, param, alpha, si, blas_functions)
    elif param['svm_type'] == 'ONE_CLASS':
        si.upper_bound = jnp.zeros(prob[0].shape[0])
        solve_one_class(prob, param, alpha, si, blas_functions)
    elif param['svm_type'] == 'EPSILON_SVR':
        si.upper_bound = jnp.zeros(prob[0].shape[0] * 2)
        solve_epsilon_svr(prob, param, alpha, si, blas_functions)
    elif param['svm_type'] == 'NU_SVR':
        si.upper_bound = jnp.zeros(prob[0].shape[0] * 2)
        solve_nu_svr(prob, param, alpha, si, blas_functions)

    status = si.solve_timed_out
    print("obj = %f, rho = %f" % (si.obj, si.rho))

    # output SVs
    nSV = 0
    nBSV = 0
    for i in range(prob[0].shape[0]):
        if abs(alpha[i]) > 0:
            nSV += 1
            if prob[1][i] > 0:
                if abs(alpha[i]) >= si.upper_bound[i]:
                    nBSV += 1
            else:
                if abs(alpha[i]) >= si.upper_bound[i]:
                    nBSV += 1

    decision_function = DecisionFunction(alpha, si.rho, si.n_iter)
    return decision_function

class SolverSolutionInfo:
    def __init__(self):
        self.upper_bound = None
        self.solve_timed_out = 0
        self.obj = 0.0
        self.rho = 0.0
        self.n_iter = 0


def sigmoid_train(l, dec_values, labels):
    prior1 = 0
    prior0 = 0
    for i in range(l):
        if labels[i] > 0:
            prior1 += 1
        else:
            prior0 += 1

    max_iter = 100
    min_step = 1e-10
    sigma = 1e-12
    eps = 1e-5
    hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
    loTarget = 1 / (prior0 + 2.0)
    t = jnp.zeros(l)
    A = 0.0
    B = jnp.log((prior0 + 1.0) / (prior1 + 1.0))
    fval = 0.0

    for i in range(l):
        if labels[i] > 0:
            t[i] = hiTarget
        else:
            t[i] = loTarget
        fApB = dec_values[i] * A + B
        if fApB >= 0:
            fval += t[i] * fApB + jnp.log(1 + jnp.exp(-fApB))
        else:
            fval += (t[i] - 1) * fApB + jnp.log(1 + jnp.exp(fApB))

    for _ in range(max_iter):
        h11 = sigma
        h22 = sigma
        h21 = 0.0
        g1 = 0.0
        g2 = 0.0

        for i in range(l):
            fApB = dec_values[i] * A + B
            if fApB >= 0:
                p = jnp.exp(-fApB) / (1.0 + jnp.exp(-fApB))
                q = 1.0 / (1.0 + jnp.exp(-fApB))
            else:
                p = 1.0 / (1.0 + jnp.exp(fApB))
                q = jnp.exp(fApB) / (1.0 + jnp.exp(fApB))
            d2 = p * q
            h11 += dec_values[i] * dec_values[i] * d2
            h22 += d2
            h21 += dec_values[i] * d2
            d1 = t[i] - p
            g1 += dec_values[i] * d1
            g2 += d1

        if jnp.abs(g1) < eps and jnp.abs(g2) < eps:
            break

        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB
        stepsize = 1

        while stepsize >= min_step:
            newA = A + stepsize * dA
            newB = B + stepsize * dB
            newf = 0.0

            for i in range(l):
                fApB = dec_values[i] * newA + newB
                if fApB >= 0:
                    newf += t[i] * fApB + jnp.log(1 + jnp.exp(-fApB))
                else:
                    newf += (t[i] - 1) * fApB + jnp.log(1 + jnp.exp(fApB))

            if newf < fval + 0.0001 * stepsize * gd:
                A = newA
                B = newB
                fval = newf
                break
            else:
                stepsize = stepsize / 2.0

        if stepsize < min_step:
            print("Line search fails in two-class probability estimates")
            break

    if _ >= max_iter:
        print("Reaching maximal iterations in two-class probability estimates")
    return A, B


def sigmoid_predict(decision_value, A, B):
    fApB = decision_value * A + B
    if fApB >= 0:
        return jnp.exp(-fApB) / (1.0 + jnp.exp(-fApB))
    else:
        return 1.0 / (1 + jnp.exp(fApB))



def multiclass_probability(k, r, p):
    max_iter = max(100, k)
    Q = jnp.zeros((k, k))
    Qp = jnp.zeros(k)
    pQp = 0
    eps = 0.005 / k

    for t in range(k):
        p[t] = 1.0 / k
        Q = jax.ops.index_update(Q, (t, t), 0)
        for j in range(t):
            Q = jax.ops.index_update(Q, (t, t), Q[t, t] + r[j, t] * r[j, t])
            Q = jax.ops.index_update(Q, (t, j), Q[j, t])
        for j in range(t + 1, k):
            Q = jax.ops.index_update(Q, (t, t), Q[t, t] + r[j, t] * r[j, t])
            Q = jax.ops.index_update(Q, (t, j), -r[j, t] * r[t, j])

    for _ in range(max_iter):
        pQp = 0
        for t in range(k):
            Qp = jnp.zeros(k)
            for j in range(k):
                Qp = jax.ops.index_add(Qp, j, Q[t, j] * p[j])
            pQp += p[t] * Qp[t]

        max_error = 0
        for t in range(k):
            error = jnp.abs(Qp[t] - pQp)
            max_error = jnp.maximum(max_error, error)

        if max_error < eps:
            break

        for t in range(k):
            diff = (-Qp[t] + pQp) / Q[t, t]
            p = jax.ops.index_add(p, t, diff)
            pQp = (pQp + diff * (diff * Q[t, t] + 2 * Qp[t])) / (1 + diff) / (1 + diff)
            for j in range(k):
                Qp = jax.ops.index_add(Qp, j, diff * Q[t, j])
                p = jax.ops.index_div(p, 1 + diff)

    if max_iter >= max_iter:
        print("Exceeds max_iter in multiclass_prob")




def svm_binary_svc_probability(prob, param, Cp, Cn, blas_functions):
    nr_fold = 5
    perm = jax.random.permutation(prob.shape[0])
    dec_values = jnp.zeros(prob.shape[0])

    for i in range(nr_fold):
        begin = i * prob.shape[0] // nr_fold
        end = (i + 1) * prob.shape[0] // nr_fold

        subprob_x = jnp.concatenate((prob[:begin], prob[end:]))
        subprob_y = jnp.concatenate((prob[:begin], prob[end:]))
        subprob_W = jnp.concatenate((prob[:begin], prob[end:]))

        p_count = jnp.sum(subprob_y > 0)
        n_count = jnp.sum(subprob_y <= 0)

        if p_count == 0 and n_count == 0:
            dec_values[perm[begin:end]] = 0
        elif p_count > 0 and n_count == 0:
            dec_values[perm[begin:end]] = 1
        elif p_count == 0 and n_count > 0:
            dec_values[perm[begin:end]] = -1
        else:
            subparam = param.copy()
            subparam.probability = 0
            subparam.C = 1.0
            subparam.nr_weight = 2
            subparam.weight_label = jnp.array([1, -1])
            subparam.weight = jnp.array([Cp, Cn])

            submodel = train(subprob_x, subprob_y, subparam, blas_functions)

            for j in range(begin, end):
                dec_values[perm[j]] = predict_values(submodel, prob[perm[j]], blas_functions) * submodel.label[0]

            destroy_model(submodel)
            destroy_param(subparam)

    probA, probB = sigmoid_train(prob.shape[0], dec_values, prob.y)
    return probA, probB



def svm_svr_probability(prob, param):
    nr_fold = 5
    ymv = jnp.zeros(prob.shape[0])
    mae = 0.0

    newparam = param.copy()
    newparam.probability = 0
    newparam.random_seed = -1

    cross_validation(prob, newparam, nr_fold, ymv)
    for i in range(prob.shape[0]):
        ymv[i] = prob[i] - ymv[i]
        mae += jnp.abs(ymv[i])
    mae /= prob.shape[0]
    std = jnp.sqrt(2 * mae * mae)
    count = 0
    mae = 0.0
    for i in range(prob.shape[0]):
        if jnp.abs(ymv[i]) > 5 * std:
            count += 1
        else:
            mae += jnp.abs(ymv[i])
    mae /= (prob.shape[0] - count)
    print(f"Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= {mae}")
    return mae



def svm_group_classes(prob):
    l = prob.l
    max_nr_class = 16
    nr_class = 0
    label = jnp.zeros(max_nr_class, dtype=int)
    count = jnp.zeros(max_nr_class, dtype=int)
    data_label = jnp.zeros(l, dtype=int)
    perm = jnp.zeros(l, dtype=int)

    for i in range(l):
        this_label = int(prob.y[i])
        for j in range(nr_class):
            if this_label == label[j]:
                count[j] += 1
                break
        else:
            if nr_class == max_nr_class:
                max_nr_class *= 2
                label = jnp.resize(label, max_nr_class)
                count = jnp.resize(count, max_nr_class)
            label[nr_class] = this_label
            count[nr_class] = 1
            nr_class += 1

    sorted_indices = jnp.argsort(label)
    label = label[sorted_indices]
    count = count[sorted_indices]

    for i in range(l):
        j = 0
        this_label = int(prob.y[i])
        while this_label != label[j]:
            j += 1
        data_label[i] = j

    start = jnp.zeros(nr_class, dtype=int)
    start[0] = 0
    for i in range(1, nr_class):
        start[i] = start[i-1] + count[i-1]

    for i in range(l):
        perm[start[data_label[i]]] = i
        start[data_label[i]] += 1

    start[0] = 0
    for i in range(1, nr_class):
        start[i] = start[i-1] + count[i-1]

    return nr_class, label, start, count, perm



def remove_zero_weight(newprob, prob):
    l = 0
    for i in range(prob['l']):
        if prob['W'][i] > 0:
            l += 1
    newprob['l'] = l
    newprob['x'] = jnp.zeros((l, prob['x'].shape[1]))
    newprob['y'] = jnp.zeros(l)
    newprob['W'] = jnp.zeros(l)

    j = 0
    for i in range(prob['l']):
        if prob['W'][i] > 0:
            newprob['x'][j] = prob['x'][i]
            newprob['y'][j] = prob['y'][i]
            newprob['W'][j] = prob['W'][i]
            j += 1

    return newprob



def svm_group_classes(prob, blas_functions):
    newprob = remove_zero_weight(prob)
    prob = newprob

    model = {}
    model['param'] = prob['param']
    model['free_sv'] = 0

    if model['param']['random_seed'] >= 0:
        jax.random.PRNGKey(model['param']['random_seed'])

    if model['param']['svm_type'] == 'ONE_CLASS' or model['param']['svm_type'] == 'EPSILON_SVR' or model['param']['svm_type'] == 'NU_SVR':
        model['nr_class'] = 2
        model['label'] = None
        model['nSV'] = None
        model['probA'] = None
        model['probB'] = None
        model['sv_coef'] = jnp.zeros((1,))

        if model['param']['probability'] and (model['param']['svm_type'] == 'EPSILON_SVR' or model['param']['svm_type'] == 'NU_SVR'):
            model['probA'] = jnp.array([svm_svr_probability(prob, model['param'], blas_functions)])

        f = svm_train_one(prob, model['param'], 0, 0, blas_functions)
        model['rho'] = jnp.array([f['rho']])
        model['n_iter'] = jnp.array([f['n_iter']])

        nSV = 0
        for i in range(prob['l']):
            if jnp.abs(f['alpha'][i]) > 0:
                nSV += 1

        model['l'] = nSV
        model['SV'] = jnp.zeros((nSV,))
        model['sv_ind'] = jnp.zeros((nSV,), dtype=int)
        model['sv_coef'][0] = jnp.zeros((nSV,))

        j = 0
        for i in range(prob['l']):
            if jnp.abs(f['alpha'][i]) > 0:
                model['SV'][j] = prob['x'][i]
                model['sv_ind'][j] = i
                model['sv_coef'][0][j] = f['alpha'][i]
                j += 1

    else:
        l = prob['l']
        nr_class = 0
        label = None
        start = None
        count = None
        perm = jnp.arange(l)

        svm_group_classes(prob, nr_class, label, start, count, perm)

        x = jnp.zeros((l,))
        W = jnp.zeros((l,))

        for i in range(l):
            x[i] = prob['x'][perm[i]]
            W[i] = prob['W'][perm[i]]

        weighted_C = jnp.zeros((nr_class,))
        for i in range(nr_class):
            weighted_C[i] = model['param']['C']

        for i in range(model['param']['nr_weight']):
            for j in range(nr_class):
                if model['param']['weight_label'][i] == label[j]:
                    break
            if j == nr_class:
                print(f"warning: class label {model['param']['weight_label'][i]} specified in weight is not found")
            else:
                weighted_C[j] *= model['param']['weight'][i]

        nonzero = jnp.zeros((l,), dtype=bool)
        svm_train_one(prob, nr_class, label, start, count, perm, weighted_C, nonzero)

        f = jnp.zeros((nr_class * (nr_class - 1) // 2,), dtype=decision_function)

        probA = None
        probB = None
        if model['param']['probability']:
            probA = jnp.zeros((nr_class * (nr_class - 1) // 2,))
            probB = jnp.zeros((nr_class * (nr_class - 1) // 2,))

        p = 0
        for i in range(nr_class):
            for j in range(i + 1, nr_class):
                sub_prob = {}
                si = start[i]
                sj = start[j]
                ci = count[i]
                cj = count[j]
                sub_prob['l'] = ci + cj
                sub_prob['x'] = jnp.zeros((sub_prob['l'],))
                sub_prob['W'] = jnp.zeros((sub_prob['l'],))
                sub_prob['y'] = jnp.zeros((sub_prob['l'],))

                for k in range(ci):
                    sub_prob['x'][k] = x[si + k]
                    sub_prob['y'][k] = 1
                    sub_prob['W'][k] = W[si + k]

                for k in range(cj):
                    sub_prob['x'][ci + k] = x[sj + k]
                    sub_prob['y'][ci + k] = -1
                    sub_prob['W'][ci + k] = W[sj + k]

                if model['param']['probability']:
                    svm_binary_svc_probability(sub_prob, model['param'], weighted_C[i], weighted_C[j], probA[p], probB[p])

                f[p] = svm_train_one(sub_prob, model['param'], weighted_C[i], weighted_C[j])
                for k in range(ci):
                    if not nonzero[si + k] and jnp.abs(f[p]['alpha'][k]) > 0:
                        nonzero[si + k] = True
                for k in range(cj):
                    if not nonzero[sj + k] and jnp.abs(f[p]['alpha'][ci + k]) > 0:
                        nonzero[sj + k] = True
                p += 1

        model['nr_class'] = nr_class
        model['label'] = label
        model['rho'] = jnp.zeros((nr_class * (nr_class - 1) // 2,))
        model['n_iter'] = jnp.zeros((nr_class * (nr_class - 1) // 2,))

        for i in range(nr_class * (nr_class - 1) // 2):
            model['rho'][i] = f[i]['rho']
            model['n_iter'][i] = f[i]['n_iter']

        if model['param']['probability']:
            model['probA'] = probA
            model['probB'] = probB
        else:
            model['probA'] = None
            model['probB'] = None

        total_sv = 0
        nz_count = jnp.zeros((nr_class,), dtype=int)
        model['nSV'] = jnp.zeros((nr_class,), dtype=int)
        for i in range(nr_class):
            nSV = 0
            for j in range(count[i]):
                if nonzero[start[i] + j]:
                    nSV += 1
                    total_sv += 1
            model['nSV'][i] = nSV
            nz_count[i] = nSV

        model['l'] = total_sv
        model['SV'] = jnp.zeros((total_sv,))
        model['sv_ind'] = jnp.zeros((total_sv,), dtype=int)

        p = 0
        for i in range(l):
            if nonzero[i]:
                model['SV'][p] = x[i]
                model['sv_ind'][p] = perm[i]
                p += 1

        nz_start = jnp.zeros((nr_class,), dtype=int)
        nz_start[0] = 0
        for i in range(1, nr_class):
            nz_start[i] = nz_start[i - 1] + nz_count[i - 1]

        model['sv_coef'] = jnp.zeros((nr_class - 1,), dtype=jnp.ndarray)
        for i in range(nr_class - 1):
            model['sv_coef'][i] = jnp.zeros((total_sv,))

        p = 0
        for i in range(nr_class):
            for j in range(i + 1, nr_class):
                si = start[i]
                sj = start[j]
                ci = count[i]
                cj = count[j]

                q = nz_start[i]
                for k in range(ci):
                    if nonzero[si + k]:
                        model['sv_coef'][j - 1][q] = f[p]['alpha'][k]
                        q += 1
                q = nz_start[j]
                for k in range(cj):
                    if nonzero[sj + k]:
                        model['sv_coef'][i][q] = f[p]['alpha'][ci + k]
                        q += 1
                p += 1

    return model


import numpy as np
from typing import List

def cross_validation(prob, param, nr_fold, blas_functions):
    l = prob.l
    perm = jax.random.permutation(l)
    fold_start = jnp.arange(0, l+1, l//nr_fold)
    target = jnp.zeros(l)

    for i in range(nr_fold):
        begin = fold_start[i]
        end = fold_start[i+1]
        subprob = create_subproblem(prob, perm, begin, end)
        submodel = train(subprob, param, blas_functions)

        if param.probability and (param.svm_type == 'C_SVC' or param.svm_type == 'NU_SVC'):
            prob_estimates = predict_probability(submodel, subprob.x, blas_functions)
            target[perm[begin:end]] = prob_estimates
        else:
            dec_values = predict_values(submodel, subprob.x, blas_functions)
            target[perm[begin:end]] = dec_values

        free_and_destroy_model(submodel)
        free_subproblem(subprob)

    return target

def get_svm_type(model):
    return model.param.svm_type

def get_nr_class(model):
    return model.nr_class

def get_labels(model):
    return model.label

def get_svr_probability(model):
    if (model.param.svm_type == 'EPSILON_SVR' or model.param.svm_type == 'NU_SVR') and model.probA is not None:
        return model.probA[0]
    else:
        raise ValueError("Model doesn't contain information for SVR probability inference")

def predict_values(model, x, blas_functions):
    if model.param.svm_type == 'ONE_CLASS' or model.param.svm_type == 'EPSILON_SVR' or model.param.svm_type == 'NU_SVR':
        sv_coef = model.sv_coef[0]
        sum = jnp.sum(sv_coef * k_function(x, model.SV, model.param, blas_functions))
        sum -= model.rho[0]
        dec_values = sum

        if model.param.svm_type == 'ONE_CLASS':
            return 1 if sum > 0 else -1
        else:
            return sum
    else:
        nr_class = model.nr_class
        l = model.l
        kvalue = k_function(x, model.SV, model.param, blas_functions)
        start = jnp.concatenate(([0], jnp.cumsum(model.nSV)[:-1]))
        vote = jnp.zeros(nr_class)

        p = 0
        for i in range(nr_class):
            for j in range(i+1, nr_class):
                sum = jnp.sum(model.sv_coef[j-1][start[i]:start[i]+model.nSV[i]] * kvalue[start[i]:start[i]+model.nSV[i]])
                sum += jnp.sum(model.sv_coef[i][start[j]:start[j]+model.nSV[j]] * kvalue[start[j]:start[j]+model.nSV[j]])
                sum -= model.rho[p]
                dec_values[p] = sum

                if dec_values[p] > 0:
                    vote[i] += 1
                else:
                    vote[j] += 1
                p += 1

        vote_max_idx = jnp.argmax(vote)
        return model.label[vote_max_idx]

def predict(model, x, blas_functions):
    dec_values = jnp.zeros(1)
    pred_result = predict_values(model, x, blas_functions)
    return pred_result

def predict_probability(model, x, blas_functions):
    if (model.param.svm_type == 'C_SVC' or model.param.svm_type == 'NU_SVC') and model.probA is not None and model.probB is not None:
        dec_values = predict_values(model, x, blas_functions)
        pairwise_prob = jnp.zeros((model.nr_class, model.nr_class))
        k = 0
        for i in range(model.nr_class):
            for j in range(i+1, model.nr_class):
                pairwise_prob[i][j] = sigmoid_predict(dec_values[k], model.probA[k], model.probB[k])
                pairwise_prob[j][i] = 1 - pairwise_prob[i][j]
                k += 1

        prob_estimates = multiclass_probability(model.nr_class, pairwise_prob)
        prob_max_idx = jnp.argmax(prob_estimates)
        return model.label[prob_max_idx]
    else:
        return predict(model, x, blas_functions)

def free_model_content(model):
    if model.free_sv and model.l > 0 and model.SV is not None:
        for i in range(model.l):
            free(model.SV[i].values)

    if model.sv_coef is not None:
        for i in range(model.nr_class-1):
            free(model.sv_coef[i])

    free(model.SV)
    free(model.sv_coef)
    free(model.sv_ind)
    free(model.rho)
    free(model.label)
    free(model.probA)
    free(model.probB)
    free(model.nSV)
    free(model.n_iter)

def free_and_destroy_model(model_ptr_ptr):
    if model_ptr_ptr is not None and model_ptr_ptr.contents is not None:
        free_model_content(model_ptr_ptr.contents)
        free(model_ptr_ptr.contents)
        model_ptr_ptr.contents = None

def destroy_param(param):
    free(param.weight_label)
    free(param.weight)

def check_parameter(prob, param):
    svm_type = param.svm_type
    if svm_type != 'C_SVC' and svm_type != 'NU_SVC' and svm_type != 'ONE_CLASS' and svm_type != 'EPSILON_SVR' and svm_type != 'NU_SVR':
        return "unknown svm type"

    kernel_type = param.kernel_type
    if kernel_type != 'LINEAR' and kernel_type != 'POLY' and kernel_type != 'RBF' and kernel_type != 'SIGMOID' and kernel_type != 'PRECOMPUTED':
        return "unknown kernel type"

    if param.gamma < 0:
        return "gamma < 0"

    if param.degree < 0:
        return "degree of polynomial kernel < 0"

    if param.cache_size <= 0:
        return "cache_size <= 0"

    if param.eps <= 0:
        return "eps <= 0"

    if svm_type == ''C_SVC' or svm_type == 'EPSILON_SVR'' or svm_type == 'NU_SVR':
        if param.C <= 0:
            return "C <= 0"

    if svm_type == 'NU_SVC' or svm_type == 'ONE_CLASS' or svm_type == 'NU_SVR':
        if param.nu <= 0 or param.nu > 1:
            return "nu <= 0 or nu > 1"

    if svm_type == 'EPSILON_SVR':
        if param.p < 0:
            return "p < 0"

    if param.shrinking != 0 and param.shrinking != 1:
        return "shrinking != 0 and shrinking != 1"

    if param.probability != 0 and param.probability != 1:
        return "probability != 0 and probability != 1"

    if param.probability == 1 and svm_type == 'ONE_CLASS':
        return "one-class SVM probability output not supported yet"

    if svm_type == 'NU_SVC':
        l = prob.l
        max_nr_class = 16
        nr_class = 0
        label = jnp.zeros(max_nr_class)
        count = jnp.zeros(max_nr_class)

        for i in range(l):
            this_label = int(prob.y[i])
            for j in range(nr_class):
                if this_label == label[j]:
                    count[j] += prob.W[i]
                    break
            if j == nr_class:
                if nr_class == max_nr_class:
                    max_nr_class *= 2
                    label = jnp.resize(label, max_nr_class)
                    count = jnp.resize(count, max_nr_class)
                label[nr_class] = this_label
                count[nr_class] = prob.W[i]
                nr_class += 1

        for i in range(nr_class):
            n1 = count[i]
            for j in range(i+1, nr_class):
                n2 = count[j]
                if param.nu * (n1 + n2) / 2 > min(n1, n2):
                    return "specified nu is infeasible"

    if svm_type == 'C_SVC' or svm_type == 'EPSILON_SVR' or svm_type == 'NU_SVR' or svm_type == 'ONE_CLASS':
        newprob = remove_zero_weight(prob)
        if newprob.l == 0:
            return "Invalid input - all samples have zero or negative weights."
        elif prob.l != newprob.l and svm_type == 'C_SVC':
            only_one_label = True
            first_label = newprob.y[0]
            for i in range(1, newprob.l):
                if newprob.y[i] != first_label:
                    only_one_label = False
                    break
            if only_one_label:
                return "Invalid input - all samples with positive weights belong to the same class."

    return None

def set_print_string_function(print_func):
    global svm_print_string
    if print_func is None:
        svm_print_string = print_string_stdout
    else:
        svm_print_string = print_func
