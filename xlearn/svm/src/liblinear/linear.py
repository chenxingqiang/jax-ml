import jax
import jax.numpy as jnp


def i:
    return i

def swap(x, y):
    t = x
    x = y
    y = t
    return x, y

def min(x, y):
    return x if x < y else y

def max(x, y):
    return x if x > y else y

def clone(src, n):
    dst = jnp.zeros(n)
    dst = jax.ops.index_update(dst, jax.ops.index[:], src)
    return dst

def Malloc(n):
    return jnp.zeros(n)

INF = float('inf')

def print_string_stdout(s):
    print(s, end='')
    print()

liblinear_print_string = print_string_stdout

def info(fmt, *args):
    buf = fmt % args
    liblinear_print_string(buf)

class l2r_lr_fun:
    def __init__(self, prob, C):
        l = prob['l']
        self.prob = prob
        self.z = jnp.zeros(l)
        self.D = jnp.zeros(l)
        self.C = C

    def fun(self, w):
        f = 0
        y = self.prob['y']
        l = self.prob['l']
        w_size = self.get_nr_variable()

        self.Xv(w, self.z)

        f += jnp.sum(w**2) / 2.0
        for i in range(l):
            yz = y[i] * self.z[i]
            if yz >= 0:
                f += self.C[i] * jnp.log(1 + jnp.exp(-yz))
            else:
                f += self.C[i] * (-yz + jnp.log(1 + jnp.exp(yz)))

        return f

    def get_nr_variable(self):
        return self.prob['n']

    def Xv(self, v, Xv):
        X = self.prob['X']
        for i in range(self.prob['l']):
            Xv[i] = jnp.dot(X[i], v)

    def grad(self, w):
        g = jnp.zeros_like(w)
        X = self.prob['X']
        y = self.prob['y']
        l = self.prob['l']
        w_size = self.get_nr_variable()

        self.Xv(w, self.z)

        for i in range(w_size):
            g[i] = w[i]

        for i in range(l):
            yz = y[i] * self.z[i]
            c = self.C[i]
            if yz >= 0:
                c = c / (1 + jnp.exp(-yz))
            else:
                c = c / (1 + jnp.exp(yz))
            g += c * X[i] * y[i]

        return g

    def Hv(self, s, Hs):
        X = self.prob['X']
        y = self.prob['y']
        l = self.prob['l']
        w_size = self.get_nr_variable()

        wa = jnp.zeros_like(s)
        self.Xv(s, wa)

        for i in range(w_size):
            Hs[i] = s[i]

        for i in range(l):
            wa[i] = self.C[i] * wa[i] * y[i] / (1 + jnp.exp(y[i] * self.z[i]))
            wa[i] = wa[i] * X[i]

        self.XTv(wa, Hs)

        for i in range(w_size):
            Hs[i] = s[i] + Hs[i]

    def XTv(self, v, XTv):
        X = self.prob['X']
        l = self.prob['l']
        w_size = self.get_nr_variable()

        for i in range(w_size):
            XTv[i] = jnp.dot(X[:, i], v)


import jax
import jax.numpy as jnp

def l2r_lr_fun_grad(w, g, prob, C):
    y = prob['y']
    l = prob['l']
    w_size = get_nr_variable(prob)

    z = jnp.zeros(l)
    D = jnp.zeros(l)
    for i in range(l):
        z[i] = 1 / (1 + jnp.exp(-y[i] * z[i]))
        D[i] = z[i] * (1 - z[i])
        z[i] = C[i] * (z[i] - 1) * y[i]

    XTv(z, g, prob)

    for i in range(w_size):
        g[i] = w[i] + g[i]

def get_nr_variable(prob):
    return prob['n']

def l2r_lr_fun_Hv(s, Hs, prob, C):
    l = prob['l']
    w_size = get_nr_variable(prob)
    wa = jnp.zeros(l)

    Xv(s, wa, prob)
    for i in range(l):
        wa[i] = C[i] * D[i] * wa[i]

    XTv(wa, Hs, prob)
    for i in range(w_size):
        Hs[i] = s[i] + Hs[i]

def Xv(v, Xv, prob):
    l = prob['l']
    X = prob['X']

    for i in range(l):
        Xv[i] = jnp.dot(X[i], v)

def XTv(v, XTv, prob):
    l = prob['l']
    w_size = get_nr_variable(prob)
    X = prob['X']

    for i in range(w_size):
        XTv[i] = 0
    for i in range(l):
        XTv += v[i] * X[i]

class l2r_l2_svc_fun:
    def __init__(self, prob, C):
        l = prob['l']
        self.prob = prob
        self.C = C
        self.z = jnp.zeros(l)
        self.D = jnp.zeros(l)
        self.I = jnp.zeros(l)
        self.sizeI = 0

    def fun(self, w):
        f = 0
        y = self.prob['y']
        l = self.prob['l']
        w_size = get_nr_variable(self.prob)

        Xv(w, self.z, self.prob)

        f += jnp.sum(w**2) / 2.0
        for i in range(w_size):
            f += self.C[i] * (1 - self.z[i])**2

        return f

    def grad(self, w, g):
        y = self.prob['y']
        l = self.prob['l']
        w_size = get_nr_variable(self.prob)

        self.sizeI = 0
        for i in range(l):
            if self.z[i] < 1:
                self.z[self.sizeI] = self.C[i] * y[i] * (self.z[i] - 1)
                self.I[self.sizeI] = i
                self.sizeI += 1
        subXTv(self.z, g, self.prob)

        for i in range(w_size):
            g[i] = w[i] + 2 * g[i]

    def Hv(self, s, Hs):
        w_size = get_nr_variable(self.prob)
        wa = jnp.zeros(self.sizeI)

        subXv(s, wa, self.prob)
        for i in range(self.sizeI):
            wa[i] = self.C[self.I[i]] * wa[i]

        subXTv(wa, Hs, self.prob)
        for i in range(w_size):
            Hs[i] = s[i] + 2 * Hs[i]

def subXv(v, Xv, prob):
    x = prob['x']

    for i in range(sizeI):
        s = x[I[i]]
        Xv[i] = 0
        while s['index'] != -1:
            Xv[i] += v[s['index'] - 1] * s['value']
            s += 1

def subXTv(v, XTv, prob):
    w_size = get_nr_variable(prob)
    x = prob['x']

    for i in range(w_size):
        XTv[i] = 0
    for i in range(sizeI):
        s = x[I[i]]
        while s['index'] != -1:
            XTv[s['index'] - 1] += v[i] * s['value']
            s += 1


class l2r_l2_svr_fun(l2r_l2_svc_fun):
    def __init__(self, prob, C, p):
        super().__init__(prob, C)
        self.p = p

    def fun(self, w):
        f = 0
        y = self.prob['y']
        l = self.prob['l']
        w_size = get_nr_variable(self.prob)
        d = jnp.zeros(l)

        Xv(w, self.z, self.prob)

        f += jnp.sum(w**2) / 2.0
        for i in range(w_size):
            f += self.C[i] * (1 - self.z[i])**2

        return f

    def grad(self, w, g):
        y = self.prob['y']
        l = self.prob['l']
        w_size = get_nr_variable(self.prob)
        d = jnp.zeros(l)

        self.sizeI = 0
        for i in range(l):
            d[i] = self.z[i] - y[i]

            if d[i] < -self.p:
                self.z[self.sizeI] = self.C[i] * (d[i] + self.p)
                self.I[self.sizeI] = i
                self.sizeI += 1
            elif d[i] > self.p:
                self.z[self.sizeI] = self.C[i] * (d[i] - self.p)
                self.I[self.sizeI] = i
                self.sizeI += 1

        subXTv(self.z, g, self.prob)

        for i in range(w_size):
            g[i] = w[i] + 2 * g[i]

class Solver_MCSVM_CS:
    def __init__(self, prob, nr_class, C, eps=0.1, max_iter=100000):
        self.w_size = prob['n']
        self.l = prob['l']
        self.nr_class = nr_class
        self.eps = eps
        self.max_iter = max_iter
        self.prob = prob
        self.B = jnp.zeros(nr_class)
        self.G = jnp.zeros(nr_class)
        self.C = jnp.zeros(prob['l'])
        for i in range(prob['l']):
            self.C[i] = prob['W'][i] * C[int(prob['y'][i])]

    def solve_sub_problem(self, A_i, yi, C_yi, active_i, alpha_new):
        D = jnp.copy(self.B[:active_i])
        if yi < active_i:
            D = jax.ops.index_update(D, yi, D[yi] + A_i * C_yi)
        D = jnp.sort(D)

        beta = D[0] - A_i * C_yi
        r = 1
        while beta < r * D[r]:
            beta += D[r]
            r += 1
        beta /= r

        for r in range(active_i):
            if r == yi:
                alpha_new = jax.ops.index_update(alpha_new, r, jnp.minimum(C_yi, (beta - self.B[r]) / A_i))
            else:
                alpha_new = jax.ops.index_update(alpha_new, r, jnp.minimum(0.0, (beta - self.B[r]) / A_i))
        return alpha_new

    def be_shrunk(self, i, m, yi, alpha_i, minG):
        bound = 0.0
        if m == yi:
            bound = self.C[i]
        if alpha_i == bound and self.G[m] < minG:
            return True
        return False

    def Solve(self, w):
        alpha = jnp.zeros(self.l * self.nr_class)
        alpha_new = jnp.zeros(self.nr_class)
        index = jnp.arange(self.l)
        QD = jnp.zeros(self.l)
        d_ind = jnp.zeros(self.nr_class)
        d_val = jnp.zeros(self.nr_class)
        alpha_index = jnp.zeros(self.nr_class * self.l)
        y_index = jnp.zeros(self.l)
        active_size = self.l
        active_size_i = jnp.zeros(self.l)
        eps_shrink = max(10.0 * self.eps, 1.0)
        start_from_all = True

        for i in range(self.l * self.nr_class):
            alpha = jax.ops.index_update(alpha, i, 0.0)

        for i in range(self.w_size * self.nr_class):
            w = jax.ops.index_update(w, i, 0.0)
        for i in range(self.l):
            for m in range(self.nr_class):
                alpha_index = jax.ops.index_update(alpha_index, i * self.nr_class + m, m)
            xi = self.prob['x'][i]
            QD = jax.ops.index_update(QD, i, 0.0)
            while xi['index'] != -1:
                val = xi['value']
                QD = jax.ops.index_add(QD, i, val * val)
                xi += 1
            active_size_i = jax.ops.index_update(active_size_i, i, self.nr_class)
            y_index = jax.ops.index_update(y_index, i, int(self.prob['y'][i]))
            index = jax.ops.index_update(index, i, i)

        iter = 0
        while iter < self.max_iter:
            stopping = -jnp.inf
            index = jax.random.permutation(index)
            for s in range(active_size):
                i = index[s]
                Ai = QD[i]
                alpha_i = alpha[i * self.nr_class: (i + 1) * self.nr_class]
                alpha_index_i = alpha_index[i * self.nr_class: (i + 1) * self.nr_class]

                if Ai > 0:
                    G = jnp.ones(self.nr_class)
                    if y_index[i] < active_size_i[i]:
                        G = jax.ops.index_update(G, y_index[i], 0.0)

                    xi = self.prob['x'][i]
                    while xi['index'] != -1:
                        w_i = w[(xi['index'] - 1) * self.nr_class: xi['index'] * self.nr_class]
                        for m in range(active_size_i[i]):
                            G = jax.ops.index_add(G, m, w_i[alpha_index_i[m]] * xi['value'])
                        xi += 1

                    minG = jnp.inf
                    maxG = -jnp.inf
                    for m in range(active_size_i[i]):
                        if alpha_i[alpha_index_i[m]] < 0 and G[m] < minG:
                            minG = G[m]
                        if G[m] > maxG:
                            maxG = G[m]
                    if y_index[i] < active_size_i[i] and alpha_i[int(self.prob['y'][i])] < self.C[i] and G[y_index[i]] < minG:
                        minG = G[y_index[i]]

                    B = G - Ai * alpha_i

                    alpha_new = self.solve_sub_problem(Ai, y_index[i], self.C[i], active_size_i[i], alpha_new)
                    nz_d = 0
                    for m in range(active_size_i[i]):
                        d = alpha_new[m] - alpha_i[alpha_index_i[m]]
                        alpha_i = jax.ops.index_update(alpha_i, alpha_index_i[m], alpha_new[m])
                        if jnp.abs(d) >= 1e-12:
                            d_ind = jax.ops.index_update(d_ind, nz_d, alpha_index_i[m])
                            d_val = jax.ops.index_update(d_val, nz_d, d)
                            nz_d += 1

                    xi = self.prob['x'][i]
                    while xi['index'] != -1:
                        w_i = w[(xi['index'] - 1) * self.nr_class: xi['index'] * self.nr_class]
                        for m in range(nz_d):
                            w_i = jax.ops.index_add(w_i, d_ind[m], d_val[m] * xi['value'])
                        xi += 1

            iter += 1
            if iter % 10 == 0:
                print(".")

            if stopping < eps_shrink:
                if stopping < self.eps and start_from_all:
                    break
                else:
                    active_size = self.l
                    active_size_i = jnp.full(self.l, self.nr_class)
                    print("*")
                    eps_shrink = max(eps_shrink / 2, self.eps)
                    start_from_all = True
            else:
                start_from_all = False

        print("\noptimization finished, #iter = ", iter)
        if iter >= self.max_iter:
            print("\nWARNING: reaching max number of iterations\n")

        v = 0.0
        nSV = 0
        for i in range(self.w_size * self.nr_class):
            v += w[i] * w[i]
        v = 0.5 * v
        for i in range(self.l * self.nr_class):
            v += alpha[i]
            if jnp.abs(alpha[i]) > 0:
                nSV += 1
        for i in range(self.l):
            v -= alpha[i * self.nr_class + int(self.prob['y'][i])]
        print("Objective value = ", v)
        print("nSV = ", nSV)

        return iter

def compare_double(a, b):
    if a > b:
        return -1
    if a < b:
        return 1
    return 0

def solve_l2r_l2_svr(prob, C, p, eps=0.1, max_iter=100000):
    l = prob['l']
    nr_class = 1
    weighted_C = jnp.zeros(l)
    for i in range(l):
        weighted_C = jax.ops.index_update(weighted_C, i, C)
    solver = Solver_MCSVM_CS(prob, nr_class, weighted_C, eps, max_iter)
    w = jnp.zeros(prob['n'])
    solver.Solve(w)
    return w


import jax
import jax.numpy as jnp

def solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, solver_type, max_iter):
    l = prob['l']
    w_size = prob['n']
    iter = 0
    C = jnp.zeros(l)
    G = jnp.zeros(l)
    alpha = jnp.zeros(l)
    y = jnp.zeros(l)
    QD = jnp.zeros(l)
    index = jnp.zeros(l)
    active_size = l
    PG = 0
    PGmax_old = jnp.inf
    PGmin_old = -jnp.inf
    PGmax_new = 0
    PGmin_new = 0
    diag = jnp.zeros(l)
    upper_bound = jnp.zeros(l)
    C_ = jnp.zeros(l)

    for i in range(l):
        if prob['y'][i] > 0:
            C_ = jax.ops.index_update(C_, i, prob['W'][i] * Cp)
        else:
            C_ = jax.ops.index_update(C_, i, prob['W'][i] * Cn)
        diag = jax.ops.index_update(diag, i, 0.5 / C_[i])
        upper_bound = jax.ops.index_update(upper_bound, i, jnp.inf)

    if solver_type == 'L2R_L1LOSS_SVC_DUAL':
        for i in range(l):
            diag = jax.ops.index_update(diag, i, 0)
            upper_bound = jax.ops.index_update(upper_bound, i, C_[i])

    for i in range(l):
        if prob['y'][i] > 0:
            y = jax.ops.index_update(y, i, +1)
        else:
            y = jax.ops.index_update(y, i, -1)

    for i in range(l):
        alpha = jax.ops.index_update(alpha, i, 0)
        QD = jax.ops.index_update(QD, i, 0)
        xi = prob['x'][i]
        while xi['index'] != -1:
            val = xi['value']
            QD = jax.ops.index_add(QD, i, val * val)
            w = jax.ops.index_add(w, xi['index'] - 1, y[i] * alpha[i] * val)
            xi += 1
        index = jax.ops.index_update(index, i, i)

    while iter < max_iter:
        PGmax_new = -jnp.inf
        PGmin_new = jnp.inf

        for i in range(active_size):
            j = i + jax.random.randint(active_size - i)
            index = jax.ops.index_update(index, i, index[j])

        for s in range(active_size):
            i = index[s]
            G = 0
            yi = y[i]
            xi = prob['x'][i]
            while xi['index'] != -1:
                G += w[xi['index'] - 1] * xi['value']
                xi += 1
            G = G * yi - 1
            C = upper_bound[i]
            G += alpha[i] * diag[i]
            PG = 0

            if alpha[i] == 0:
                if G > PGmax_old:
                    active_size -= 1
                    index = jax.ops.index_update(index, s, index[active_size])
                    s -= 1
                    continue
                elif G < 0:
                    PG = G
            elif alpha[i] == C:
                if G < PGmin_old:
                    active_size -= 1
                    index = jax.ops.index_update(index, s, index[active_size])
                    s -= 1
                    continue
                elif G > 0:
                    PG = G
            else:
                PG = G

            PGmax_new = jnp.maximum(PGmax_new, PG)
            PGmin_new = jnp.minimum(PGmin_new, PG)

            if jnp.abs(PG) > 1.0e-12:
                alpha_old = alpha[i]
                alpha = jax.ops.index_update(alpha, i, jnp.minimum(jnp.maximum(alpha[i] - G / QD[i], 0.0), C))
                d = (alpha[i] - alpha_old) * yi
                xi = prob['x'][i]
                while xi['index'] != -1:
                    w = jax.ops.index_add(w, xi['index'] - 1, d * xi['value'])
                    xi += 1

        iter += 1
        if iter % 10 == 0:
            print(".")

        if PGmax_new - PGmin_new <= eps:
            if active_size == l:
                break
            else:
                active_size = l
                print("*")
                PGmax_old = jnp.inf
                PGmin_old = -jnp.inf
                continue

        PGmax_old = PGmax_new
        PGmin_old = PGmin_new
        if PGmax_old <= 0:
            PGmax_old = jnp.inf
        if PGmin_old >= 0:
            PGmin_old = -jnp.inf

    print("\noptimization finished, #iter = ", iter)
    if iter >= max_iter:
        print("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n")

    v = 0
    nSV = 0
    for i in range(w_size):
        v += w[i] * w[i]
    v = 0.5 * v
    for i in range(l):
        v += alpha[i]
        if jnp.abs(alpha[i]) > 0:
            nSV += 1
    for i in range(l):
        v -= alpha[i] * (alpha[i] * diag[i] - 2)
        if alpha[i] > 0:
            nSV += 1

    print("Objective value = ", v)
    print("nSV = ", nSV)
    return iter

def solve_l2r_l1l2_svr(prob, w, eps, Cp, Cn, solver_type, max_iter):
    l = prob['l']
    C = prob['C']
    p = prob['p']
    w_size = prob['n']
    eps = prob['eps']
    iter = 0
    active_size = l
    index = jnp.arange(l)
    beta = jnp.zeros(l)
    QD = jnp.zeros(l)
    y = prob['y']
    G = jnp.zeros(l)
    H = jnp.zeros(l)
    Gmax_old = jnp.inf
    Gmax_new = 0
    Gnorm1_new = 0
    Gnorm1_init = -1.0
    Cp = prob['Cp']
    Cn = prob['Cn']
    lambda_ = jnp.zeros(l)
    upper_bound = jnp.zeros(l)
    C_ = jnp.zeros(l)

    for i in range(l):
        C_ = jax.ops.index_update(C_, i, C)
        lambda_ = jax.ops.index_update(lambda_, i, 0.5 / C_[i])
        upper_bound = jax.ops.index_update(upper_bound, i, jnp.inf)

    if solver_type == 'L2R_L1LOSS_SVR_DUAL':
        for i in range(l):
            lambda_ = jax.ops.index_update(lambda_, i, 0)
            upper_bound = jax.ops.index_update(upper_bound, i, C_[i])

    for i in range(l):
        beta = jax.ops.index_update(beta, i, 0)

    for i in range(w_size):
        w = jax.ops.index_update(w, i, 0)

    for i in range(l):
        xi = prob['x'][i]
        while xi['index'] != -1:
            val = xi['value']
            QD = jax.ops.index_add(QD, i, val * val)
            w = jax.ops.index_add(w, xi['index'] - 1, beta[i] * val)
            xi += 1
        index = jax.ops.index_update(index, i, i)

    while iter < max_iter:
        Gmax_new = 0
        Gnorm1_new = 0

        for s in range(active_size):
            i = index[s]
            G = -y[i] + lambda_[i] * beta[i]
            H = QD[i] + lambda_[i]
            xi = prob['x'][i]
            while xi['index'] != -1:
                ind = xi['index'] - 1
                val = xi['value']
                G += val * w[ind]
                xi += 1

            Gp = G + p
            Gn = G - p
            violation = 0

            if beta[i] == 0:
                if Gp < 0:
                    violation = -Gp
                elif Gn > 0:
                    violation = Gn
                elif Gp > Gmax_old and Gn < -Gmax_old:
                    active_size -= 1
                    index = jax.ops.index_update(index, s, index[active_size])
                    s -= 1
                    continue
            elif beta[i] >= upper_bound[i]:
                if Gp > 0:
                    violation = Gp
                elif Gp < -Gmax_old:
                    active_size -= 1
                    index = jax.ops.index_update(index, s, index[active_size])
                    s -= 1
                    continue
            elif beta[i] <= -upper_bound[i]:
                if Gn < 0:
                    violation = -Gn
                elif Gn > Gmax_old:
                    active_size -= 1
                    index = jax.ops.index_update(index, s, index[active_size])
                    s -= 1
                    continue
            elif beta[i] > 0:
                violation = jnp.abs(Gp)
            else:
                violation = jnp.abs(Gn)

            Gmax_new = jnp.maximum(Gmax_new, violation)
            Gnorm1_new += violation

            if jnp.abs(d) < 1.0e-12:
                continue

            beta_old = beta[i]
            beta = jax.ops.index_update(beta, i, jnp.minimum(jnp.maximum(beta[i] + d, -upper_bound[i]), upper_bound[i]))
            d = beta[i] - beta_old

            if d != 0:
                xi = prob['x'][i]
                while xi['index'] != -1:
                    w = jax.ops.index_add(w, xi['index'] - 1, d * xi['value'])
                    xi += 1

        if iter == 0:
            Gnorm1_init = Gnorm1_new
        iter += 1
        if iter % 10 == 0:
            print(".")

        if Gnorm1_new <= eps * Gnorm1_init:
            if active_size == l:
                break
            else:
                active_size = l
                print("*")
                Gmax_old = jnp.inf
                continue

        Gmax_old = Gmax_new

    print("\noptimization finished, #iter = ", iter)
    if iter >= max_iter:
        print("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n")

    v = 0
    nSV = 0
    for i in range(w_size):
        v += w[i] * w[i]
    v = 0.5 * v
    for i in range(l):
        v += p * jnp.abs(beta[i]) - y[i] * beta[i] + 0.5 * lambda_[i] * beta[i] * beta[i]
        if beta[i] != 0:
            nSV += 1

    print("Objective value = ", v)
    print("nSV = ", nSV)
    return iter


def solve_l1r_l2_svc(prob_col, w, eps, Cp, Cn, max_iter):
    l = prob_col['l']
    w_size = prob_col['n']
    active_size = w_size
    max_num_linesearch = 20
    sigma = 0.01
    d_old = 0
    d_diff = 0
    loss_old = 0
    loss_new = 0
    appxcond = 0
    cond = 0
    Gmax_old = jnp.inf
    Gmax_new = 0
    Gnorm1_new = 0
    Gnorm1_init = -1.0
    iter = 0

    index = jnp.arange(w_size)
    y = jnp.zeros(l)
    b = jnp.zeros(l)
    xj_sq = jnp.zeros(w_size)
    x = jnp.zeros(w_size)

    C = jnp.zeros(l)

    for i in range(l):
        b = jax.ops.index_update(b, i, 1)
        if prob_col['y'][i] > 0:
            y = jax.ops.index_update(y, i, 1)
            C = jax.ops.index_update(C, i, prob_col['W'][i] * Cp)
        else:
            y = jax.ops.index_update(y, i, -1)
            C = jax.ops.index_update(C, i, prob_col['W'][i] * Cn)

    for j in range(w_size):
        index = jax.ops.index_update(index, j, j)
        xj_sq = jax.ops.index_update(xj_sq, j, 0)
        x = prob_col['x'][j]
        while x['index'] != -1:
            ind = x['index'] - 1
            x = jax.ops.index_update(x, 'value', x['value'] * y[ind])
            val = x['value']
            b = jax.ops.index_add(b, ind, -w[j] * val)
            xj_sq = jax.ops.index_add(xj_sq, j, C[GETI(ind)] * val * val)
            x += 1

    while iter < max_iter:
        Gmax_new = 0
        Gnorm1_new = 0

        for s in range(active_size):
            j = index[s]
            G_loss = 0
            H = 0

            x = prob_col['x'][j]
            while x['index'] != -1:
                ind = x['index'] - 1
                if b[ind] > 0:
                    val = x['value']
                    tmp = C[GETI(ind)] * val
                    G_loss += -tmp * b[ind]
                    H += tmp * val
                x += 1
            G_loss *= 2

            G = G_loss
            H *= 2
            H = jnp.maximum(H, 1e-12)

            Gp = G + 1
            Gn = G - 1
            violation = 0
            if w[j] == 0:
                if Gp < 0:
                    violation = -Gp
                elif Gn > 0:
                    violation = Gn
                elif Gp > Gmax_old / l and Gn < -Gmax_old / l:
                    active_size -= 1
                    index = jax.ops.index_update(index, s, index[active_size])
                    s -= 1
                    continue
            elif w[j] > 0:
                violation = jnp.abs(Gp)
            else:
                violation = jnp.abs(Gn)

            Gmax_new = jnp.maximum(Gmax_new, violation)
            Gnorm1_new += violation

            if Gp < H * w[j]:
                d = -Gp / H
            elif Gn > H * w[j]:
                d = -Gn / H
            else:
                d = -w[j]

            if jnp.abs(d) < 1.0e-12:
                continue

            delta = jnp.abs(w[j] + d) - jnp.abs(w[j]) + G * d
            d_old = 0
            num_linesearch = 0
            while num_linesearch < max_num_linesearch:
                d_diff = d_old - d
                cond = jnp.abs(w[j] + d) - jnp.abs(w[j]) - sigma * delta

                appxcond = xj_sq[j] * d * d + G_loss * d + cond
                if appxcond <= 0:
                    x = prob_col['x'][j]
                    while x['index'] != -1:
                        ind = x['index'] - 1
                        b = jax.ops.index_add(b, ind, d_diff * x['value'])
                        x += 1
                    break

                if num_linesearch == 0:
                    loss_old = 0
                    loss_new = 0
                    x = prob_col['x'][j]
                    while x['index'] != -1:
                        ind = x['index'] - 1
                        if b[ind] > 0:
                            loss_old += C[GETI(ind)] * b[ind] * b[ind]
                        b_new = b[ind] + d_diff * x['value']
                        b = jax.ops.index_update(b, ind, b_new)
                        if b_new > 0:
                            loss_new += C[GETI(ind)] * b_new * b_new
                        x += 1
                else:
                    loss_new = 0
                    x = prob_col['x'][j]
                    while x['index'] != -1:
                        ind = x['index'] - 1
                        b_new = b[ind] + d_diff * x['value']
                        b = jax.ops.index_update(b, ind, b_new)
                        if b_new > 0:
                            loss_new += C[GETI(ind)] * b_new * b_new
                        x += 1

                cond = cond + loss_new - loss_old
                if cond <= 0:
                    break
                else:
                    d_old = d
                    d *= 0.5
                    delta *= 0.5

            w = jax.ops.index_add(w, j, d)

            if num_linesearch >= max_num_linesearch:
                print("#")
                for i in range(l):
                    b = jax.ops.index_update(b, i, 1)

                for i in range(w_size):
                    if w[i] == 0:
                        continue
                    x = prob_col['x'][i]
                    while x['index'] != -1:
                        b = jax.ops.index_add(b, x['index'] - 1, -w[i] * x['value'])
                        x += 1

        if iter == 0:
            Gnorm1_init = Gnorm1_new
        iter += 1
        if iter % 10 == 0:
            print(".")

        if Gnorm1_new <= eps * Gnorm1_init:
            if active_size == w_size:
                break
            else:
                active_size = w_size
                print("*")
                Gmax_old = jnp.inf
                continue

        Gmax_old = Gmax_new

    print("\noptimization finished, #iter = ", iter)
    if iter >= max_iter:
        print("\nWARNING: reaching max number of iterations\n")

    v = 0
    nnz = 0
    for j in range(w_size):
        x = prob_col['x'][j]
        while x['index'] != -1:
            x = jax.ops.index_update(x, 'value', x['value'] * prob_col['y'][x['index'] - 1])
            x += 1
        if w[j] != 0:
            v += jnp.abs(w[j])
            nnz += 1
    for j in range(l):
        if b[j] > 0:
            v += C[j] * b[j] * b[j]

    print("Objective value = ", v)
    print("#nonzeros/#features = ", nnz, "/", w_size)
    return iter

def solve_l1r_lr(prob_col, w, eps, Cp, Cn, max_newton_iter):
    l = prob_col.l
    w_size = prob_col.n
    j, s, newton_iter, iter = 0, 0, 0, 0
    max_iter = 1000
    max_num_linesearch = 20
    active_size = 0
    QP_active_size = 0
    QP_no_change = 0

    nu = 1e-12
    inner_eps = 1
    sigma = 0.01
    w_norm, w_norm_new = 0, 0
    z, G, H = 0, 0, 0
    Gnorm1_init = -1.0
    Gmax_old = jnp.inf
    Gmax_new, Gnorm1_new = 0, 0
    QP_Gmax_old = jnp.inf
    QP_Gmax_new, QP_Gnorm1_new = 0, 0
    delta, negsum_xTd, cond = 0, 0, 0

    index = jnp.arange(w_size)
    y = jnp.zeros(l)
    Hdiag = jnp.zeros(w_size)
    Grad = jnp.zeros(w_size)
    wpd = jnp.zeros(w_size)
    xjneg_sum = jnp.zeros(w_size)
    xTd = jnp.zeros(l)
    exp_wTx = jnp.zeros(l)
    exp_wTx_new = jnp.zeros(l)
    tau = jnp.zeros(l)
    D = jnp.zeros(l)

    for j in range(w_size):
        w[j] = 0

    for j in range(l):
        if prob_col.y[j] > 0:
            y[j] = 1
            C = prob_col.W[j] * Cp
        else:
            y[j] = -1
            C = prob_col.W[j] * Cn

        exp_wTx[j] = 0

    for j in range(w_size):
        w_norm += jnp.abs(w[j])
        wpd[j] = w[j]
        index[j] = j
        xjneg_sum[j] = 0
        x = prob_col.x[j]
        while x.index != -1:
            ind = x.index - 1
            val = x.value
            exp_wTx[ind] += w[j] * val
            if y[ind] == -1:
                xjneg_sum[j] += C[GETI(ind)] * val
            x = x.next

    for j in range(l):
        exp_wTx[j] = jnp.exp(exp_wTx[j])
        tau_tmp = 1 / (1 + exp_wTx[j])
        tau[j] = C[j] * tau_tmp
        D[j] = C[j] * exp_wTx[j] * tau_tmp * tau_tmp

    while newton_iter < max_newton_iter:
        Gmax_new = 0
        Gnorm1_new = 0
        active_size = w_size

        for s in range(active_size):
            j = index[s]
            Hdiag[j] = nu
            Grad[j] = 0

            tmp = 0
            x = prob_col.x[j]
            while x.index != -1:
                ind = x.index - 1
                Hdiag[j] += x.value * x.value * D[ind]
                tmp += x.value * tau[ind]
                x = x.next
            Grad[j] = -tmp + xjneg_sum[j]

            Gp = Grad[j] + 1
            Gn = Grad[j] - 1
            violation = 0
            if w[j] == 0:
                if Gp < 0:
                    violation = -Gp
                elif Gn > 0:
                    violation = Gn
                elif Gp > Gmax_old / l and Gn < -Gmax_old / l:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    s -= 1
            elif w[j] > 0:
                violation = jnp.abs(Gp)
            else:
                violation = jnp.abs(Gn)

            Gmax_new = jnp.maximum(Gmax_new, violation)
            Gnorm1_new += violation

        if newton_iter == 0:
            Gnorm1_init = Gnorm1_new

        if Gnorm1_new <= eps * Gnorm1_init or QP_no_change >= 10:
            break

        QP_no_change += 1
        iter = 0
        QP_Gmax_old = jnp.inf
        QP_active_size = active_size

        xTd = jnp.zeros(l)

        while iter < max_iter:
            QP_Gmax_new = 0
            QP_Gnorm1_new = 0

            for j in range(QP_active_size):
                i = j + bounded_rand_int(QP_active_size - j)
                index[i], index[j] = index[j], index[i]

            for s in range(QP_active_size):
                j = index[s]
                H = Hdiag[j]

                x = prob_col.x[j]
                G = Grad[j] + (wpd[j] - w[j]) * nu
                while x.index != -1:
                    ind = x.index - 1
                    G += x.value * D[ind] * xTd[ind]
                    x = x.next

                Gp = G + 1
                Gn = G - 1
                violation = 0
                if wpd[j] == 0:
                    if Gp < 0:
                        violation = -Gp
                    elif Gn > 0:
                        violation = Gn
                    elif Gp > QP_Gmax_old / l and Gn < -QP_Gmax_old / l:
                        QP_active_size -= 1
                        index[s], index[QP_active_size] = index[QP_active_size], index[s]
                        s -= 1
                elif wpd[j] > 0:
                    violation = jnp.abs(Gp)
                else:
                    violation = jnp.abs(Gn)

                if Gp < H * wpd[j]:
                    z = -Gp / H
                elif Gn > H * wpd[j]:
                    z = -Gn / H
                else:
                    z = -wpd[j]

                if jnp.abs(z) < 1.0e-12:
                    continue
                z = jnp.minimum(np.maximum(z, -10.0), 10.0)

                QP_no_change = 0
                QP_Gmax_new = jnp.maximum(QP_Gmax_new, violation)
                QP_Gnorm1_new += violation

                wpd[j] += z

                x = prob_col.x[j]
                while x.index != -1:
                    ind = x.index - 1
                    xTd[ind] += x.value * z
                    x = x.next

            iter += 1

            if QP_Gnorm1_new <= inner_eps * Gnorm1_init:
                if QP_active_size == active_size:
                    break
                else:
                    QP_active_size = active_size
                    QP_Gmax_old = jnp.inf
                    continue

            QP_Gmax_old = QP_Gmax_new

        if iter >= max_iter:
            print("WARNING: reaching max number of inner iterations\n")

        delta = 0
        w_norm_new = 0
        for j in range(w_size):
            delta += Grad[j] * (wpd[j] - w[j])
            if wpd[j] != 0:
                w_norm_new += jnp.abs(wpd[j])
        delta += (w_norm_new - w_norm)

        negsum_xTd = 0
        for i in range(l):
            if y[i] == -1:
                negsum_xTd += C[i] * xTd[i]

        for num_linesearch in range(max_num_linesearch):
            cond = w_norm_new - w_norm + negsum_xTd - sigma * delta

            for i in range(l):
                exp_xTd = jnp.exp(xTd[i])
                exp_wTx_new[i] = exp_wTx[i] * exp_xTd
                cond += C[i] * jnp.log((1 + exp_wTx_new[i]) / (exp_xTd + exp_wTx_new[i]))

            if cond <= 0:
                w_norm = w_norm_new
                for j in range(w_size):
                    w[j] = wpd[j]
                for i in range(l):
                    exp_wTx[i] = exp_wTx_new[i]
                    tau_tmp = 1 / (1 + exp_wTx[i])
                    tau[i] = C[i] * tau_tmp
                    D[i] = C[i] * exp_wTx[i] * tau_tmp * tau_tmp
                break
            else:
                w_norm_new = 0
                for j in range(w_size):
                    wpd[j] = (w[j] + wpd[j]) * 0.5
                    if wpd[j] != 0:
                        w_norm_new += jnp.abs(wpd[j])
                delta *= 0.5
                negsum_xTd *= 0.5
                xTd *= 0.5

        if num_linesearch >= max_num_linesearch:
            exp_wTx = jnp.zeros(l)
            for i in range(w_size):
                if w[i] == 0:
                    continue
                x = prob_col.x[i]
                while x.index != -1:
                    exp_wTx[x.index - 1] += w[i] * x.value
                    x = x.next

            exp_wTx = jnp.exp(exp_wTx)

        if iter == 1:
            inner_eps *= 0.25

        newton_iter += 1
        Gmax_old = Gmax_new

        print("iter %3d  #CD cles %d\n" % (newton_iter, iter))

    print("=========================\n")
    print("optimization finished, #iter = %d\n" % newton_iter)
    if newton_iter >= max_newton_iter:
        print("WARNING: reaching max number of iterations\n")

    v = 0
    nnz = 0
    for j in range(w_size):
        if w[j] != 0:
            v += jnp.abs(w[j])
            nnz += 1
    for j in range(l):
        if y[j] == 1:
            v += C[j] * jnp.log(1 + 1 / exp_wTx[j])
        else:
            v += C[j] * jnp.log(1 + exp_wTx[j])

    print("Objective value = %lf\n" % v)
    print("#nonzeros/#features = %d/%d\n" % (nnz, w_size))




def train_one(prob, param, w, Cp, Cn, blas_functions):
    eps = param['eps']
    max_iter = param['max_iter']
    pos = jnp.sum(prob['y'] > 0)
    neg = prob['l'] - pos

    primal_solver_tol = eps * max(min(pos, neg), 1) / prob['l']

    fun_obj = None
    if param['solver_type'] == 'L2R_LR':
        C = jnp.where(prob['y'] > 0, prob['W'] * Cp, prob['W'] * Cn)
        fun_obj = l2r_lr_fun(prob, C)
        tron_obj = TRON(fun_obj, primal_solver_tol, max_iter, blas_functions)
        tron_obj.set_print_string(liblinear_print_string)
        n_iter = tron_obj.tron(w)
    elif param['solver_type'] == 'L2R_L2LOSS_SVC':
        C = jnp.where(prob['y'] > 0, prob['W'] * Cp, prob['W'] * Cn)
        fun_obj = l2r_l2_svc_fun(prob, C)
        tron_obj = TRON(fun_obj, primal_solver_tol, max_iter, blas_functions)
        tron_obj.set_print_string(liblinear_print_string)
        n_iter = tron_obj.tron(w)
    elif param['solver_type'] == 'L2R_L2LOSS_SVC_DUAL':
        n_iter = solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL, max_iter)
    elif param['solver_type'] == 'L2R_L1LOSS_SVC_DUAL':
        n_iter = solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL, max_iter)
    elif param['solver_type'] == 'L1R_L2LOSS_SVC':
        prob_col = transpose(prob)
        n_iter = solve_l1r_l2_svc(prob_col, w, primal_solver_tol, Cp, Cn, max_iter)
    elif param['solver_type'] == 'L1R_LR':
        prob_col = transpose(prob)
        n_iter = solve_l1r_lr(prob_col, w, primal_solver_tol, Cp, Cn, max_iter)
    elif param['solver_type'] == 'L2R_LR_DUAL':
        n_iter = solve_l2r_lr_dual(prob, w, eps, Cp, Cn, max_iter)
    elif param['solver_type'] == 'L2R_L2LOSS_SVR':
        C = prob['W'] * param['C']
        fun_obj = l2r_l2_svr_fun(prob, C, param['p'])
        tron_obj = TRON(fun_obj, param['eps'], max_iter, blas_functions)
        tron_obj.set_print_string(liblinear_print_string)
        n_iter = tron_obj.tron(w)
    elif param['solver_type'] == 'L2R_L1LOSS_SVR_DUAL':
        n_iter = solve_l2r_l1l2_svr(prob, w, param, L2R_L1LOSS_SVR_DUAL, max_iter)
    elif param['solver_type'] == 'L2R_L2LOSS_SVR_DUAL':
        n_iter = solve_l2r_l1l2_svr(prob, w, param, L2R_L2LOSS_SVR_DUAL, max_iter)
    else:
        raise ValueError("Unknown solver_type")

    return n_iter

def remove_zero_weight(newprob, prob):
    l = jnp.sum(prob['W'] > 0)
    newprob['l'] = l
    newprob['x'] = prob['x'][prob['W'] > 0]
    newprob['y'] = prob['y'][prob['W'] > 0]
    newprob['W'] = prob['W'][prob['W'] > 0]

def train(prob, param, blas_functions):
    newprob = prob.copy()
    remove_zero_weight(newprob, prob)
    prob = newprob
    l = prob['l']
    n = prob['n']
    w_size = prob['n']
    model_ = {'nr_feature': None, 'param': param, 'bias': prob['bias']}

    if prob['bias'] >= 0:
        model_['nr_feature'] = n - 1
    else:
        model_['nr_feature'] = n

    if check_regression_model(model_):
        model_['w'] = jnp.zeros(w_size)
        model_['n_iter'] = jnp.zeros(1, dtype=int)
        model_['nr_class'] = 2
        model_['label'] = None
        model_['n_iter'][0] = train_one(prob, param, model_['w'], 0, 0, blas_functions)
    else:
        nr_class, label, start, count, perm = group_classes(prob)
        model_['nr_class'] = nr_class
        model_['label'] = label

        weighted_C = jnp.full(nr_class, param['C'])
        for i in range(param['nr_weight']):
            for j in range(nr_class):
                if param['weight_label'][i] == label[j]:
                    weighted_C[j] *= param['weight'][i]

        x = prob['x'][perm]
        sub_prob = {'l': l, 'n': n, 'x': x, 'y': prob['y'][perm], 'W': prob['W'][perm]}

        if param['solver_type'] == 'MCSVM_CS':
            model_['w'] = jnp.zeros(w_size * nr_class)
            model_['n_iter'] = jnp.zeros(1, dtype=int)
            sub_prob['y'] = jnp.array([i for i in range(nr_class) for _ in range(count[i])])
            Solver = Solver_MCSVM_CS(sub_prob, nr_class, weighted_C, param['eps'])
            model_['n_iter'][0] = Solver.Solve(model_['w'])
        else:
            if nr_class == 2:
                model_['w'] = jnp.zeros(w_size)
                model_['n_iter'] = jnp.zeros(1, dtype=int)
                e0 = start[0] + count[0]
                sub_prob['y'][:e0] = -1
                sub_prob['y'][e0:] = 1
                model_['n_iter'][0] = train_one(sub_prob, param, model_['w'], weighted_C[1], weighted_C[0], blas_functions)
            else:
                model_['w'] = jnp.zeros(w_size * nr_class)
                w = jnp.zeros(w_size)
                model_['n_iter'] = jnp.zeros(nr_class, dtype=int)
                for i in range(nr_class):
                    si = start[i]
                    ei = si + count[i]
                    sub_prob['y'][:si] = -1
                    sub_prob['y'][si:ei] = 1
                    sub_prob['y'][ei:] = -1
                    model_['n_iter'][i] = train_one(sub_prob, param, w, weighted_C[i], param['C'], blas_functions)
                    model_['w'][jnp.arange(w_size) * nr_class + i] = w

    return model_

def remove_zero_weight(newprob, prob):
    l = jnp.sum(prob['W'] > 0)
    newprob['l'] = l
    newprob['x'] = prob['x'][prob['W'] > 0]
    newprob['y'] = prob['y'][prob['W'] > 0]
    newprob['W'] = prob['W'][prob['W'] > 0]

def get_nr_feature(model_):
    return model_['nr_feature']

def get_nr_class(model_):
    return model_['nr_class']

def get_labels(model_):
    if model_['label'] is not None:
        return model_['label']

def get_n_iter(model_):
    labels = model_['nr_class']
    if labels == 2:
        labels = 1
    if model_['n_iter'] is not None:
        return model_['n_iter']

def free_model_content(model_ptr):
    if model_ptr['w'] is not None:
        del model_ptr['w']
    if model_ptr['label'] is not None:
        del model_ptr['label']
    if model_ptr['n_iter'] is not None:
        del model_ptr['n_iter']

def free_and_destroy_model(model_ptr_ptr):
    model_ptr = model_ptr_ptr[0]
    if model_ptr is not None:
        free_model_content(model_ptr)
        del model_ptr

def destroy_param(param):
    if param['weight_label'] is not None:
        del param['weight_label']
    if param['weight'] is not None:
        del param['weight']

def check_parameter(prob, param):
    if param['eps'] <= 0:
        return "eps <= 0"

    if param['C'] <= 0:
        return "C <= 0"

    if param['p'] < 0:
        return "p < 0"

    solver_types = ['L2R_LR', 'L2R_L2LOSS_SVC_DUAL', 'L2R_L2LOSS_SVC', 'L2R_L1LOSS_SVC_DUAL', 'MCSVM_CS',
                    'L1R_L2LOSS_SVC', 'L1R_LR', 'L2R_LR_DUAL', 'L2R_L2LOSS_SVR', 'L2R_L2LOSS_SVR_DUAL',
                    'L2R_L1LOSS_SVR_DUAL']
    if param['solver_type'] not in solver_types:
        return "unknown solver type"

    return None

def check_regression_model(model_):
    return model_['param']['solver_type'] in ['L2R_L2LOSS_SVR', 'L2R_L1LOSS_SVR_DUAL', 'L2R_L2LOSS_SVR_DUAL']

def set_print_string_function(print_func):
    if print_func is None:
        liblinear_print_string = print_string_stdout
    else:
        liblinear_print_string = print_func
