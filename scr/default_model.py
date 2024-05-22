import settings as s
from random import uniform
import colorama
import memory_worker as mem

colorama.init()

####################################################### Params ##################################################################

w_arr = []                      #
a = s.a                     # Параметры
b = s.b                     # системы
c = s.c                     #
t_max = s.t_max             # Время интегрирования

k_str = s.k_str             # Число агентов в одной строке
k_col = s.k_col             # Число агентов в одном столбце
k_elements = k_str * k_col  # Число агентов 
k = s.k                       # Число уравнений для одного агента (всегда 3)
T = s.T
radius = s.radius
tau = s.tau

####################################################### Params ##################################################################

# Создаем массив частот(w) для всех агентов
def generate_w_arr(k_elements, _range=[0.93, 1.07]):
    w_arr = []
    for i in range(k_elements):
        w_arr.append(round(uniform(_range[0], _range[1]), 3))
    return w_arr

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r, T_):
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
def func_dx(index, r, connect_f=default_f, _T=s.T, w_arr = w_arr):
    return - w_arr[index] * r[index*k + 1] - r[index*k + 2] + connect_f(index, r, _T)


# По y.
def func_dy(index, r, connect_f=default_f, _T=s.T, w_arr = w_arr):
    return w_arr[index] * r[index*k] + a * r[index*k + 1] + connect_f(index, r, _T)


# По z
def func_dz(index, r, _T, connect_f=default_f):
    return b + r[index*k + 2] * (r[index*k] - c) + connect_f(index, r, _T)


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
        dz = tau_ * func_dz(i, r, T)

        res_arr.append(dx)
        res_arr.append(dy)
        res_arr.append(dz)

    return res_arr