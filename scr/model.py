import agents_grid as ag
from config import settings as s
from random import uniform
import colorama

colorama.init()

####################################################### Params ##################################################################

w = []                      #
a = s.a                     # Параметры
b = s.b                     # системы
c = s.c                     #
t_max = s.t_max             # Время интегрирования

k_str = s.k_str             # Число агентов в одной строке
k_col = s.k_col             # Число агентов в одном столбце
k_elements = k_str * k_col  # Число агентов 
k = s.k                       # Число уравнений для одного агента (всегда 3)
T = s.T

small_animation = s.small_animation
full_animation = s.full_animation
need_save_last_state = s.need_save_last_state

radius = s.radius           # Радиус связи
min_radius = s.min_radius   # Минимальный радиус, меньше которого появляются дополнительные эффекты
undeleted_elems = []

border_radius = s.stopping_border_radius
border_center = s.stopping_border_center
stopping_borded_work = s.stopping_borded_work

####################################################### Params ##################################################################

# Создаем массив частот(w) для всех агентов
def generate_w_arr(k_elements, _range=[0.93, 1.07]):
    w_arr = []
    for i in range(k_elements):
        w_arr.append(round(uniform(_range[0], _range[1]), 3))
    return w_arr

# Удаляет агента (изменяет его НУ) и удаляет соответствующий индекс из массива неудаленных элементов
def delete_agent(item, arr, undeleted_elems, value=10000):
    undeleted_elems.remove(item)

    return arr, undeleted_elems

def find_min_elem_array_in_range(arr, start, stop):
    # Находим первый НЕудаленный элемент из промежутка [start, stop] и запоминаем его номер в массиве
    el = 0
    for i in range(len(undeleted_elems)):
        if undeleted_elems[i] < start:
            el = i
        else:
            return el + 1
        
    return -1
    

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r, T_):
    return 0


# Функция связи по y. Последовательное движение цепочки агентов
def func_connect_y(index, r, _T):
    summ = 0
    for i in range(k_elements):
        summ += d(_T, radius, r[i*k], r[index*k], r[i*k+1], r[index*k+1]) * (r[i*k + 1] - r[index*k + 1])
    return summ

def func_connect_y_3dim(index, r, _T):
    summ = 0
    for i in range(k_elements):
        summ += d_3dim(_T, radius, r[i*k], r[index*k], r[i*k+1], r[index*k+1], r[i*k+2], r[index*k+2]) \
                * (r[i*k + 1] - r[index*k + 1])
    return summ

def func_connect_y_VDP(index, r, _T):

    summ = 0
    for j in range(k_elements):
        # summ += d_3dim(_T, radius, r[j*k], r[index*k], r[j*k+1], r[index*k+1], r[j*k+2], r[index*k+2]) \
        #         * (r[j*k + 1] - r[index*k + 1])
        summ += d_3dim(_T, radius, r[j*k], r[-3], r[j*k+1], r[-2], r[j*k+2], r[-1]) \
                * (r[-2] - r[index*k + 1])
        
    # Связь сетка 
    summ_grid = func_connect_y_grid(index, r, _T)
    return summ + summ_grid

def func_connect_y_grid(index, r, _T):
    summ = 0
    summ2 = 0
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

    if start > undeleted_elems[-1]:
        return 0
    
    el = find_min_elem_array_in_range(undeleted_elems, start, stop)

    for j in range(start, stop):
        if undeleted_elems[el] != j:
            continue

        el += 1
        summ += d_3dim(_T, radius, r[j*k], r[index*k], r[j*k+1], r[index*k+1], r[j*k+2], r[index*k+2]) \
                * (r[j*k + 1] - r[index*k + 1])
        
        # if index != j:
        #     summ2 += d_3dim(_T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
        #             / (r[index * k + 1] - r[j * k + 1])
        
        if el == len(undeleted_elems):
            return summ
        
    # print('debag', index, n_string, start, stop)

    return summ

# Функция связи по x. Параллельное движение цепочки агентов
def func_connect_x(index, r, _T):
    summ1, summ2 = 0, 0
    for j in range(k_elements):   
        if j != index:
            summ1 += d(_T, radius, r[j*k], r[index*k], r[j*k+1], r[index*k+1]) * (r[j*k] - r[index*k])
            summ2 += d(_T, radius, r[j*k], r[index*k], r[j*k+1], r[index*k+1]) / (r[index*k] - r[j*k])
            
    return summ1 + summ2


def func_connect_x_3dim(index, r, _T):
    summ1, summ2 = 0, 0
    for j in range(k_elements):
        
        if j != index:
            summ1 += d_3dim(_T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
                     * (r[j * k] - r[index * k])
            summ2 += d_3dim(_T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
                     / (r[index * k] - r[j * k])

    return summ1 + summ2

def func_connect_x_grid(index, r, _T):
    n_string = index // k_str
    start = n_string * k_str
    stop = (n_string + 1) * k_str
    summ1, summ2 = 0, 0

    # Находим первый НЕудаленный элемент из промежутка [start, stop] и запоминаем его номер в массиве
    el = find_min_elem_array_in_range(undeleted_elems, start, stop)
    if el == -1:
        return 0

    for j in range(start, stop):
        if undeleted_elems[el] != j:
            continue

        el += 1
        if j != index:
            summ1 += d_3dim(_T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
                        * (r[j * k] - r[index * k])
            summ2 += d_3dim(_T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
                        / (r[index * k] - r[j * k])
            
        if el == len(undeleted_elems):
            return summ1 + summ2

    return summ1 + summ2

# Функция для проверки, какие соседи находятся ближе минимального радиуса
def connect_min_radius(index, r, min_radius, undeleted_elems):

    tmp = 0
    for j in undeleted_elems:
        if index == j:
            continue

        # Если два элемента ближе минимального радиуса
        if d_3dim(1, min_radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) != 0:
            # Нужно удалить эти элементы
            r, undeleted_elems = delete_agent(j, r, undeleted_elems)
            tmp = 1
    
    if tmp == 1:
        r, undeleted_elems = delete_agent(index, r, undeleted_elems)
    return r, undeleted_elems, tmp

# Функция включения связи между двумя агентами
def d(_T, _radius, x_i, x_j, y_i, y_j):
    if (x_i - x_j)**2 + (y_i - y_j)**2 < _radius**2:
        return _T
    else:
        return 0

def d_3dim(_T, _radius, x_i, x_j, y_i, y_j, z_i, z_j):
    if (x_i - x_j)**2 + (y_i - y_j)**2 + (z_i - z_j)**2 < _radius**2:
        return _T
    else:
        return 0


# Функции правой части
# По x
def func_dx(index, r, connect_f=default_f, _T=s.T, w_arr = w):
    if len(undeleted_elems) == 1:
        return - w_arr[index] * r[index*k + 1] - r[index*k + 2]
    return - w_arr[index] * r[index*k + 1] - r[index*k + 2] + connect_f(index, r, _T)


# По y.
def func_dy(index, r, connect_f=default_f, _T=s.T, w_arr = w):
    if len(undeleted_elems) == 1:
        return w_arr[index] * r[index*k] + a * r[index*k + 1]
    return w_arr[index] * r[index*k] + a * r[index*k + 1] + connect_f(index, r, _T)


# По z
def func_dz(index, r, _T, connect_f=default_f):
    if len(undeleted_elems) == 1:
        return b + r[index*k + 2] * (r[index*k] - c)
    return b + r[index*k + 2] * (r[index*k] - c) + connect_f(index, r, _T)

####################################################### Rossler ##################################################################

def func_rossler_3_dim(t, r):
    global k_elements
    res_arr = []

    for i in range(k_elements):
        # x_i = r[i*k]
        # y_i = r[i*k + 1]
        # z_i = r[i*k + 2]

        dx = func_dx(i, r, func_connect_x_grid, T, w)
        dy = func_dy(i, r, func_connect_y_grid, T, w)
        dz = func_dz(i, r, T)

        res_arr.append(dx)
        res_arr.append(dy)
        res_arr.append(dz)

    return res_arr

last_t = []
remove = False
last_update_time = 0
def func_rossler_del_elems(t, r, k_elements, w_arr, undeleted_elems_, T_):
    # Вывод в последней строке текущее время интегрирования
    global remove, last_update_time
    if round(t, 2) % 2 == 0:
        if remove == True:
            print(f'Current integrate time: {round(t, 2)};', f'last update time: {ag.hms_now()}')
            remove = False
        else: 
            print(f'\033[F\033[KCurrent integrate time: {round(t, 2)};', f'last update time: {ag.hms_now()}')
        last_update_time = ag.hms_now(type = 'm')

    # global last_m
    # if ag.hms_now(type = 'm') - last_m > 0:
    #     print('one more minute', f'last_update_time {last_update_time}', ag.hms_now(type = 'm') - last_update_time, ag.hms_now(type = 'm') - last_update_time > 5)
    #     last_m = ag.hms_now(type = 'm')

    global last_t
    if t > 1:
        if ag.hms_now(type = 'm') - last_update_time >= 5:
            tyme_diff = ag.hms_now(type = 'm') - last_update_time
            print(f'System broken. h = {t - last_t[-1000]}, {tyme_diff}')
    last_t.append(t)


    global undeleted_elems
    undeleted_elems = undeleted_elems_

    res_arr = []

    if len(undeleted_elems) == 0:
        return [0 for i in range(k_elements * k)]

    counter = 0
    for i in undeleted_elems:
        if counter < i:
            for tmp_index in range(i - counter):
                res_arr.append(0)
                res_arr.append(0)
                res_arr.append(0)
            counter = i

        checker = 0
        # Добавляем проверку ближнего радиуса
        if min_radius > 0:
            r, undeleted_elems, checker = connect_min_radius(i, r, min_radius, undeleted_elems)

        # Проверка барьера
        if stopping_borded_work == True:
            if (r[i*k] - border_center[0])**2 + (r[i*k+1] - border_center[1])**2 >= border_radius**2:
                checker = 1
                undeleted_elems.remove(i)
                print(f'remove {i}', undeleted_elems, f'time {t}', f'x {r[i*k]}', f'y {r[i*k+1]}', (r[i*k] - border_center[0])**2 + (r[i*k+1] - border_center[1])**2)
                remove = True

                # Проверка на все удаленные элементы из-за борьера
                if len(undeleted_elems) == 0:
                    return [0 for i in range(k_elements * k)]
            
        if checker == 0:
            dx = func_dx(i, r, func_connect_x_grid, T_, w_arr=w_arr)
            dy = func_dy(i, r, func_connect_y_grid, T_, w_arr=w_arr)
            dz = func_dz(i, r, T_)

            res_arr.append(dx)
            res_arr.append(dy)
            res_arr.append(dz)
        else:
            res_arr.append(0)
            res_arr.append(0)
            res_arr.append(0)

        counter += 1

    while counter < k_elements:
        res_arr.append(0)
        res_arr.append(0)
        res_arr.append(0)
        counter += 1

    return res_arr

def function_rossler_and_VanDerPol(t, r, k_elements, w_arr, mu, W):
    res_arr = []

    for i in range(k_elements):
        # x_i = r[i*k]
        # y_i = r[i*k + 1]
        # z_i = r[i*k + 2]

        dx = func_dx(i, r, func_connect_x_grid, T, w_arr)
        dy = func_dy(i, r, func_connect_y_VDP, T, w_arr)
        dz = func_dz(i, r, T)

        res_arr.append(dx)
        res_arr.append(dy)
        res_arr.append(dz)

    # Van der Pol
    # X = r[-3]
    # Y = r[-2]
    # Z = r[-1]
    res_arr.append(- r[-2])
    res_arr.append(W * r[-3] + mu * (1 - r[-3]**2) * r[-2])
    res_arr.append(0)

    return res_arr

