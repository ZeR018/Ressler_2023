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

####################################################### Grid functions ##################################################################

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r, T_, p):
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
def func_dx(i, r, connect_f=default_f, _T=s.T, w_arr = w_arr, connect_f_inh = default_f):
    return - w_arr[i] * r[i*k + 1] - r[i*k + 2] + connect_f(i, r, _T, 'x') + connect_f_inh(i, r, _T, 'x')


# По y.
def func_dy(i, r, connect_f=default_f, _T=s.T, w_arr = w_arr, connect_f_inh = default_f):
    return w_arr[i] * r[i*k] + a * r[i*k + 1] + connect_f(i, r, _T, 'y') + connect_f_inh(i, r, _T, 'y')


# По z
def func_dz(i, r, connect_f=default_f, _T = s.T, connect_f_inh = default_f):
    return b + r[i*k + 2] * (r[i*k] - c) + connect_f(i, r, _T, 'z') + connect_f_inh(i, r, _T, 'z')


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
def d(_T, _radius, x_i, x_j, y_i, y_j):
    if (x_i - x_j)**2 + (y_i - y_j)**2 < _radius**2:
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

def f_connect_st(i, r, _T, perem = 'y'):
    if perem == 'z':
        p_shift = 2
    if perem == 'x':
        p_shift = 0
    else:
        p_shift = 1


    summ = 0
    for j in range(k_elements):
        if j != i:
            summ += d(_T, radius, r[j*k], r[i*k], r[j*k+1], r[i*k+1]) * (r[j*k + p_shift] - r[i*k + p_shift])
    return summ

def f_connect_inh(i, r, _T, perem = 'y'):
    if perem == 'z':
        p_shift = 2
    if perem == 'x':
        p_shift = 0
    else:
        p_shift = 1

    summ = 0
    for j in range(k_elements):
        if j != i:
            summ += d(_T, radius, r[j*k], r[i*k], r[j*k+1], r[i*k+1]) / (r[i*k + p_shift] - r[j*k + p_shift])
    return summ

def func_rossler_2_dim(t, r, w_arr_, a_, tau_ = tau):
    global k_elements, w_arr, a
    print(k_elements)
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

def func_rossler_2_dim_params_maker(k_elements_, couplings = (False, True, False), couplings_inh = (False, False, False)):
    f_dx_coup = f_connect_st if couplings[0] else default_f
    f_dy_coup = f_connect_st if couplings[1] else default_f
    f_dz_coup = f_connect_st if couplings[2] else default_f
    f_dx_coup_inh = f_connect_inh if couplings_inh[0] else default_f
    f_dy_coup_inh = f_connect_inh if couplings_inh[1] else default_f
    f_dz_coup_inh = f_connect_inh if couplings_inh[2] else default_f

    def func_rossler_2_dim_params(t, r, w_arr_, a_, tau_ = tau):
        global k_elements, w_arr, a
        k_elements = k_elements_
        w_arr = w_arr_
        a = a_
        res_arr = []

        for i in range(k_elements):
            # x_i = r[i*k]
            # y_i = r[i*k + 1]
            # z_i = r[i*k + 2]

            dx = tau_ * func_dx(i, r, f_dx_coup, T, w_arr, connect_f_inh=f_dx_coup_inh)
            dy = tau_ * func_dy(i, r, f_dy_coup, T, w_arr, connect_f_inh=f_dy_coup_inh)
            dz = tau_ * func_dz(i, r, f_dz_coup, T, connect_f_inh=f_dz_coup_inh)

            res_arr.append(dx)
            res_arr.append(dy)
            res_arr.append(dz)

        return res_arr
    return func_rossler_2_dim_params

####################################################### events ##################################################################

phi = [[] for i in range(k_elements)]
omega = [[] for i in range(k_elements)]
step = 300
def synchronization_event(t, r, w_arr_, a_, tau_):
    max_omega_diff = 0.25

    omega_range_last = []
    for agent in range(k_elements):
        x = r[agent*k]
        y = r[agent*k + 1]
        z = r[agent*k + 2]
        w = w_arr[agent]
        phi_i = np.arctan(( w * x + a * y) / (- w * y - z))
        phi[agent].append(phi_i)
        
        omega_i_dyddx = - (w*x + a*y) * (-w**2*x-w*a*y-b-z*(x-c))
        omega_i_ddydx = (-w*y-z) * (-w**2*y-w*z+a*w*x+a**2*y)

        omega_i_zn = (w*y+z)**2 + (w*x + a*y)**2
        omega[agent].append((omega_i_ddydx + omega_i_dyddx)/omega_i_zn)

        if len(omega) > step:
            omega_range_last.append(np.mean(omega[agent][-step]))
        else:
            return -1
        
        # Проверить, что расстояние между всеми агентами меньше 1.5r
        x0 = r[0]
        y0 = r[1]
        z0 = r[2]
        dist = (np.sqrt((x - x0)**2+ \
                (y - y0)**2 + (z - z0)**2))
        if dist > 1.5 * s.radius:
            return -1
    
    if max(omega_range_last) - min(omega_range_last) > max_omega_diff:
        return -1
    
    print("sync!!!!", t[-1])
    return 1
        


    