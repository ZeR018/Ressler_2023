import settings as s
from random import uniform
import colorama
import memory_worker as mem
import numpy as np

colorama.init()

####################################################### Params ##################################################################

w_arr = []                      #
a = s.a                     # Параметры
b = s.b                     # системы
c = s.c                     #
t_max = s.t_max             # Время интегрирования

k_str = s.k_str             # Число агентов в одной строке
k_col = s.k_col             # Число агентов в одном столбце
k_elements = 0              # Число агентов 
k = s.k                     # Число уравнений для одного агента (всегда 3)
T = s.T
radius = s.radius
tau = s.tau

if s.exps_type == 'grid':
    k_elements = k_str * k_col
else:
    k_elements = s.k_elements

min_radius = 1.

####################################################### Grid functions ##################################################################

def calc_norm_inf(i, r):
    return max(abs(r[i*k]), abs(r[i*k+1]), abs(r[i*k+2]))

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r, T_, perem = 'y', k = s.k, k_elements = k_elements, radius=radius):
    return 0

def func_connect_y_grid(index, r, _T):
    summ = 0
    start, stop = 0, 0

    n_string = index // k_str
    if n_string == 0:
        start = 0
        stop = k_str
    else:
        start = k_str * (n_string-1)

    if n_string == k_col - 1:
        stop = k_elements
    else:
        stop = (n_string + 2) * k_str

    for i in range(start, stop):
        summ += d_3dim(_T, radius, r[i*k], r[index*k], r[i*k+1], r[index*k+1], r[i*k+2], r[index*k+2]) \
                * (r[i*k + 1] - r[index*k + 1])

    # print('debag', index, n_string, start, stop)

    return summ

def func_connect_x_grid(index, r, _T):
    n_string = index // k_str
    start = n_string * k_str
    stop = (n_string + 1) * k_str
    summ1, summ2 = 0, 0

    for j in range(start, stop):
        if j != index:
            summ1 += d_3dim(_T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
                        * (r[j * k] - r[index * k])
            summ2 += d_3dim(_T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
                        / (r[index * k] - r[j * k])

    return summ1 + summ2

def d_3dim(_T, _radius, x_i, x_j, y_i, y_j, z_i, z_j):
    if (x_i - x_j)**2 + (y_i - y_j)**2 + (z_i - z_j)**2 < _radius**2:
        return _T
    else:
        return 0

# Функции правой части
# По x
def func_dx(i, r, connect_f=default_f, _T=s.T, w_arr = w_arr, connect_f_rep = default_f):
    return - w_arr[i] * r[i*k + 1] - r[i*k + 2] + connect_f(i, r, _T, perem='x', k_elements=k_elements) + connect_f_rep(i, r, _T, 'x')


# По y.
def func_dy(i, r, connect_f=default_f, _T=s.T, w_arr = w_arr, connect_f_rep = default_f):
    return w_arr[i] * r[i*k] + a * r[i*k + 1] + connect_f(i, r, _T, perem='y', k_elements=k_elements) + connect_f_rep(i, r, _T, 'y')


# По z
def func_dz(i, r, connect_f=default_f, _T = s.T, connect_f_rep = default_f):
    return b + r[i*k + 2] * (r[i*k] - c) + connect_f(i, r, _T, perem='z', k_elements=k_elements) + connect_f_rep(i, r, _T, 'z')

def func_rossler_3_dim(t, r, w_arr_, a_, tau_ = tau):
    # print(f'\033[F\033[KCurrent integrate time: {round(t, 1)};', f'last update time: {mem.hms_now()}')
        
    global k_elements, w_arr, a
    w_arr = w_arr_
    a = a_
    res_arr = []

    for i in range(k_elements):
        # x_i = r[i*k]
        # y_i = r[i*k + 1]
        # z_i = r[i*k + 2]

        dx = tau_ * func_dx(i, r, func_connect_x_grid, T, w_arr)
        dy = tau_ * func_dy(i, r, func_connect_y_grid, T, w_arr)
        dz = tau_ * func_dz(i, r, _T = T)

        res_arr.append(dx)
        res_arr.append(dy)
        res_arr.append(dz)

    return res_arr

####################################################### Posledovatelnoye ##################################################################

# Функция включения связи между двумя агентами
def d(_T, _radius, x_i, x_j, y_i, y_j, min_radius = 0.):
    dist = (x_i - x_j)**2 + (y_i - y_j)**2
    if dist < _radius**2:
        return _T
    else:
        return 0

# Функция связи по x. Параллельное движение цепочки агентов
def f_connect_x_repulsive(i, r, _T, perem = 'x'):
    summ1, summ2 = 0, 0
    for j in range(k_elements):   
        if j != i:
            summ1 += d(_T, radius, r[j*k], r[i*k], r[j*k+1], r[i*k+1]) * (r[j*k] - r[i*k])
            summ2 += d(_T, radius, r[j*k], r[i*k], r[j*k+1], r[i*k+1]) / (r[i*k] - r[j*k])
            
    return summ1 + summ2

def f_connect_st(i, r, _T, perem = 'y', k=s.k, k_elements=k_elements, radius = s.radius):
    if perem == 'z':
        p_shift = 2
    elif perem == 'x':
        p_shift = 0
    else:
        p_shift = 1

    summ = 0
    for j in range(k_elements):
        if j != i:
            summ += d(_T, radius, r[j*k], r[i*k], r[j*k+1], r[i*k+1], min_radius=min_radius) * (r[j*k + p_shift] - r[i*k + p_shift])
    return summ

def f_connect_rep(i, r, _T, perem = 'y'):
    ind = i * k

    if perem == 'z':
        p_shift = 2
    elif perem == 'x':
        p_shift = 0
    else:
        p_shift = 1

    summ = 0
    for j in range(k_elements):
        jnd = j * k
        if j != i:
            summ += d(_T, radius, r[jnd], r[ind], r[jnd+1], r[ind+1]) / (r[ind + p_shift] - r[jnd + p_shift])
    return summ

def func_rossler_2_dim(t, r, w_arr_, a_, tau_ = tau):
    # print(f'\033[F\033[KCurrent integrate time: {round(t, 1)};', f'last update time: {mem.hms_now()}')
    global k_elements, w_arr, a
    w_arr = w_arr_
    a = a_
    res_arr = []

    for i in range(k_elements):
        # x_i = r[i*k]
        # y_i = r[i*k + 1]
        # z_i = r[i*k + 2]

        dx = tau_ * func_dx(i, r, default_f, T, w_arr)
        dy = tau_ * func_dy(i, r, f_connect_st, T, w_arr)
        dz = tau_ * func_dz(i, r, _T = T)

        res_arr.append(dx)
        res_arr.append(dy)
        res_arr.append(dz)

    return res_arr

# def func_rossler_2_dim_params_maker(k_elements_, couplings = (False, True, False), couplings_inh = (False, False, False)):
#     f_dx_coup = f_connect_st if couplings[0] else default_f
#     f_dy_coup = f_connect_st if couplings[1] else default_f
#     f_dz_coup = f_connect_st if couplings[2] else default_f
#     f_dx_coup_inh = f_connect_inh if couplings_inh[0] else default_f
#     f_dy_coup_inh = f_connect_inh if couplings_inh[1] else default_f
#     f_dz_coup_inh = f_connect_inh if couplings_inh[2] else default_f

#     def func_rossler_2_dim_params(t, r, w_arr_, a_, tau_ = tau):
#         # if round(t, 3) % 2 == 0:
#         #     print(t)
#         global k_elements, w_arr, a
#         k_elements = k_elements_
#         w_arr = w_arr_
#         a = a_
#         res_arr = []

#         for i in range(k_elements):
#             # x_i = r[i*k]
#             # y_i = r[i*k + 1]
#             # z_i = r[i*k + 2]

#             # Связь по z
#             if couplings[2]:
#                 if calc_norm_inf(i, r) > 10000:
#                     # elements.remove(i)
#                     # print('remove', i)
#                     # continue
#                     res_arr.append(0)
#                     res_arr.append(0)
#                     res_arr.append(0)
#                     continue

#             dx = tau_ * func_dx(i, r, f_dx_coup, T, w_arr, connect_f_inh=f_dx_coup_inh)
#             dy = tau_ * func_dy(i, r, f_dy_coup, T, w_arr, connect_f_inh=f_dy_coup_inh)
#             dz = tau_ * func_dz(i, r, f_dz_coup, T, connect_f_inh=f_dz_coup_inh)

#             res_arr.append(dx)
#             res_arr.append(dy)
#             res_arr.append(dz)

#         return res_arr
#     return func_rossler_2_dim_params

def func_rossler_2_dim_params_maker(k_elements_, couplings = (False, True, False), T_ = T, couplings_rep = (False, False, False)):
    f_dx_coup = f_connect_st if couplings[0] else default_f
    f_dy_coup = f_connect_st if couplings[1] else default_f
    f_dz_coup = f_connect_st if couplings[2] else default_f

    f_dx_coup_rep = f_connect_rep if couplings_rep[0] else default_f
    f_dy_coup_rep = f_connect_rep if couplings_rep[1] else default_f
    f_dz_coup_rep = f_connect_rep if couplings_rep[2] else default_f

    global T
    T = T_

    def func_rossler_2_dim_params(t, r, w_arr_, a_, tau_ = tau):
        # if round(t, 2) % 2 == 0:
        #     print(t)
        global k_elements, w_arr, a
        k_elements = k_elements_
        w_arr = w_arr_
        a = a_
        res_arr = []

        for i in range(k_elements):
            # x_i = r[i*k]
            # y_i = r[i*k + 1]
            # z_i = r[i*k + 2]

            # Связь по z
            if couplings_rep[2]:
                if calc_norm_inf(i, r) > 10000:
                    # elements.remove(i)
                    # continue
                    res_arr.append(0)
                    res_arr.append(0)
                    res_arr.append(0)
                    continue

            dx = tau_ * func_dx(i, r, f_dx_coup, T, w_arr, connect_f_rep=f_dx_coup_rep)
            dy = tau_ * func_dy(i, r, f_dy_coup, T, w_arr, connect_f_rep=f_dy_coup_rep)
            dz = tau_ * func_dz(i, r, f_dz_coup, T, connect_f_rep=f_dz_coup_rep)

            res_arr.append(dx)
            res_arr.append(dy)
            res_arr.append(dz)

        return res_arr
    return func_rossler_2_dim_params

############################################################# Lorenz ################################################################

def func_dx_lorenz(i, X, l, c, connect_f = default_f):
    return l.sigma * (X[i*l.k + 1] - X[i*l.k]) + connect_f(i, X, c.T, 'x', k=l.k, k_elements=c.k_elements, radius=c.radius)

def func_dy_lorenz(i, X, l, c, connect_f = default_f):
    return l.r[i] * X[i*l.k] - X[i*l.k + 1] - X[i*l.k] * X[i*l.k + 2] + connect_f(i, X, c.T, 'y', k=l.k, k_elements=c.k_elements, radius=c.radius)

def func_dz_lorenz(i, X, l, c, connect_f = default_f):
    return - l.b * X[i*l.k + 2] + X[i*l.k] * X[i * l.k + 1] + connect_f(i, X, c.T, 'z', k=l.k, k_elements=c.k_elements, radius=c.radius)

class Lorenz_params:
    def __init__(self, sigma = 10, b = 8/3, r = [166.1, 166.12], k = 3):
        self.k = k
        self.sigma = sigma
        self.b = b
        # self.d = d
        self.r = r

class Coup_params:
    def __init__(self, k_elements, radius, T, couplings = (False, False, False)):
        self.T = T
        self.k_elements = k_elements
        self.couplings = couplings
        self.radius = radius

def func_lorenz_params(l: Lorenz_params, c: Coup_params, tau = 1):
    f_dx_coup = f_connect_st if c.couplings[0] else default_f
    f_dy_coup = f_connect_st if c.couplings[1] else default_f
    f_dz_coup = f_connect_st if c.couplings[2] else default_f

    def func_lorenz(t, X):
        res_arr = []

        for i in range(c.k_elements):
            dx = tau * func_dx_lorenz(i, X, l, c, connect_f=f_dx_coup)
            dy = tau * func_dy_lorenz(i, X, l, c, connect_f=f_dy_coup)
            dz = tau * func_dz_lorenz(i, X, l, c, connect_f=f_dz_coup)

            res_arr.append(dx)
            res_arr.append(dy)
            res_arr.append(dz)
        
        return res_arr
    
    return func_lorenz

    