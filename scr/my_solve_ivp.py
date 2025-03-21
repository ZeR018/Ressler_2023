from matplotlib import pyplot as plt
import numpy as np

# Шаг методом Эйлера
def euler(f, t, h, x_arr, size, params):
    x_next_arr = []
    try:
        x_next_arr = [x_arr[i] + f(t, x_arr, params)[i] * h for i in range(size)]
    except IndexError:
        print('Размерности полученной системы и начальных условий не совпадают')
        return -1
    return x_next_arr

# Шаг методом РК4
def RK4(f, t, h, x_arr, size, params):
    if params:
        k1 = f(t, x_arr, params)
        k2 = f(t + h/2., [x_arr[i] + k1[i]/2 * h for i in range(size)], params)
        k3 = f(t + h/2., [x_arr[i] + k2[i]/2 * h for i in range(size)], params)
        k4 = f(t + h, [x_arr[i] + k3[i] * h for i in range(size)], params)
    else:
        k1 = f(t, x_arr)
        k2 = f(t + h/2., [x_arr[i] + k1[i]/2 * h for i in range(size)])
        k3 = f(t + h/2., [x_arr[i] + k2[i]/2 * h for i in range(size)])
        k4 = f(t + h, [x_arr[i] + k3[i] * h for i in range(size)])
    x_next_arr = [x_arr[i] + (k1[i]+ 2 * k2[i] + 2 * k3[i] + k4[i])/6. * h for i in range(size)]

    return x_next_arr

# Метод численного интегрирования для обоих методов. Выбор метода осуществляется
# добавлением параметра "method"
def my_solve(f, time, IC, h, params = None, method=RK4):
    if type(IC) == int or type(IC) == float:
        IC = [IC]
    size = len(IC)
    x_arr = [IC]
    x_next_arr = []
    tMax = time[1]
    t_arr = np.arange(time[0], time[1], h)
    h2 = h/2.
    ksi = []

    for t in t_arr:

        try:
            x_next_arr = method(f, t, h, x_arr[-1], size, params)
        except IndexError:
            print('Размерности полученной системы и начальных условий не совпадают')
            return -1

        x_arr.append(x_next_arr)

        # Подсчет ошибки через половинный шаг
        x_next_arr = method(f, t, h2, x_arr[-2], size, params)
        x_next_arr = method(f, t, h2, x_next_arr, size, params)
        ksi.append(size_of_vectors(x_arr[-1], x_next_arr, size))

    x_arr = np.array(x_arr)
    x_arr = x_arr.T

    t_arr = np.append(t_arr, tMax)

    return t_arr, x_arr, ksi

# Поиск длины между двумя точками
def size_of_vectors(arr1, arr2, size = '0'):
    if size == '0':
        size = len(arr1)

    sum = 0
    for i in range(size):
        sum += (arr1[i] - arr2[i])**2

    return np.sqrt(sum)