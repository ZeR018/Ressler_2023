from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import time
from config import settings as s
from random import uniform
from datetime import datetime
import main as m
import os
from matplotlib.animation import ArtistAnimation

w = []                      #
a = s.a                     # Параметры
b = s.b                     # системы
c = s.c                     #
t_max = 150

k_str = 5                   # Число агентов в одной строке
k_col = 5                   # Число агентов в одном столбце
k_elements = k_str * k_col  # Число агентов 
k = 3                       # Число уравнений для одного агента (всегда 3)
T = 0.15

radius = s.radius           # Радиус связи
T_attractive = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1]           # Сила притягивающей связи

T_repulsive = [0.3, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.1, 0.3]            # Сила отталкивающей связи


# Функции синхронизации

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r):
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

def func_connect_x_grid_10(index, r, _T):
    summ1, summ2 = 0, 0
    start, stop = 0, 0
    if index < 5:
        start = 0
        stop = 5
    else:
        start = 5
        stop = k_elements

    for j in range(start, stop):
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

    for j in range(start, stop):
        if j != index:
            summ1 += d_3dim(_T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
                        * (r[j * k] - r[index * k])
            summ2 += d_3dim(_T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
                        / (r[index * k] - r[j * k])

    return summ1 + summ2


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


def d_3dim(_T, _radius, x_i, x_j, y_i, y_j, z_i, z_j):
    if (x_i - x_j)**2 + (y_i - y_j)**2 + (z_i - z_j)**2 < _radius**2:
        return _T
    else:
        return 0


# Функции правой части
# По x
def func_dx(index, r, connect_f=default_f, _T=s.T):
    return - w[index] * r[index*k + 1] - r[index*k + 2] + connect_f(index, r, _T)


# По y.
def func_dy(index, r, connect_f=default_f, _T=s.T):
    return w[index] * r[index*k] + a * r[index*k + 1] + connect_f(index, r, _T)


# По z
def func_dz(index, r, connect_f=default_f):
    return b + r[index*k + 2] * (r[index*k] - c) + connect_f(index, r)


def func_rossler_3_dim(t, r):
    global k_elements
    res_arr = []

    for i in range(k_elements):
        # x_i = r[i*k]
        # y_i = r[i*k + 1]
        # z_i = r[i*k + 2]

        dx = func_dx(i, r, func_connect_x_grid, T)
        dy = func_dy(i, r, func_connect_y_grid, T)
        dz = func_dz(i, r)

        res_arr.append(dx)
        res_arr.append(dy)
        res_arr.append(dz)

    return res_arr


# Создаем массив частот(w) для всех агентов
def generate_w_arr(k_elements, _range=[0.93, 1.07]):
    w_arr = []
    for i in range(k_elements):
        w_arr.append(round(uniform(_range[0], _range[1]), 3))
    return w_arr


# Возвращает время с точностью до секунды
def hms_now():
    return str(datetime.now().time())[:-7]


# Сохраняет начальные условия и массив частот(w) в указанный файл
def save_IC_and_w(IC, w, path, _k_elements = k_elements):
    with open(path, 'w') as f:
        print(_k_elements, file=f)
        for i in range(_k_elements):
            print(IC[i*k], IC[i*k+1], IC[i*k+2], file=f)

        print(w, file=f)
        print('T:', T, file=f)
        print('k_col:', k_col, 'k_str:', k_str, file=f)


# Считывает сохраненные НУ и w из указанного файла
def read_IC_and_w(path):
    IC = []
    with open(path, 'r') as f:
        f_data = f.readlines()

    size = int(f_data[0])

    for i in range(1, size+1):
        line = f_data[i].split()
        IC.append(float(line[0]))
        IC.append(float(line[1]))
        IC.append(float(line[2]))

    w_str = f_data[-1][1:-2].split()
    w = [float(i[:-1]) for i in w_str]
    w[-1] = float(w_str[-1])

    return IC, w


# Сохраняет данные, полученные интегрированием в указанный файл
def save_integration_data(integration_data, path, _k_elements = k_elements):
    xs, ys, zs, ts = integration_data
    
    size = len(xs[0])
    with open(path, 'w') as f:
        for i in range(size):
            for agent in range(_k_elements):
                print(xs[agent][i], ys[agent][i], zs[agent][i], file=f)
        for i in range(size):
            print(ts[i], file=f)
        print(_k_elements, file=f)
        print(size, file=f)


# Считывает данные интегрирования из указанного файла
def read_integration_data(path):
       
    with open(path, 'r') as f:
        f_data = f.readlines()

    _k_elements = int(f_data[-2])
    size = int(f_data[-1])

    xs, ys, zs, ts = [], [], [], []
    for i in range(_k_elements):
        xs.append([])
        ys.append([])
        zs.append([])
    for i in range(size):
        for agent in range(_k_elements):
            line = f_data[i * _k_elements + agent].split()
            xs[agent].append(float(line[0]))
            ys[agent].append(float(line[1]))
            zs[agent].append(float(line[2]))
    for i in range(size*_k_elements, size*(_k_elements+1)):
        ts.append(float(f_data[i]))

    return [xs, ys, zs, ts]


# Сохраняет данные интегрирования, НУ, w, и все необходимые графики
def save_data(integration_data, IC, w, figs_arr, fig_names_arr):
    date = str(datetime.now().date())
    time = hms_now().replace(':', '.')

    new_dir = s.grid_experiments_path + date + ' ' + time
    os.mkdir(new_dir)

    save_IC_and_w(IC, w,new_dir + '/IC.txt')
    save_integration_data(integration_data, new_dir + '/integration_data.txt')
    
    for i in range(len(figs_arr)):
        figs_arr[i].savefig(new_dir + '/' + fig_names_arr[i] + '.png')

    # Для картинок
    data_dir = new_dir + '/data'
    os.mkdir(data_dir)

    return new_dir, data_dir

def make_frames_grid_agents(xs_arr, ys_arr, plot_colors, _k_elements = k_elements, frames_step = 25):

    fig = plt.figure(figsize=[12,12])
    ax = fig.add_subplot()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    fig.suptitle('Сетка мобильных агентов')

    # Кол-во кадров
    num_frames = len(xs_arr[0])
    frames = []

    for i in range(0, num_frames, frames_step):
        frame = []

        for agent in range(_k_elements):
            
            line = 0
            if i < 500:
                line, = ax.plot(xs_arr[agent][:i], ys_arr[agent][:i], color=plot_colors[agent])
            else:
                line, = ax.plot(xs_arr[agent][i-500:i], ys_arr[agent][i-500:i], color=plot_colors[agent])
            frame.append(line)

            point = ax.scatter(xs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])
            frame.append(point)

        frames.append(frame)
    return frames, fig


def make_colors(_k_elements):
    res_col_arr = []
    color_index = 1

    color_step = 1 / _k_elements
    for i in range(_k_elements):
        res_col_arr.append((round(1 - color_index, 5), round(color_index, 5), round(1 - color_index, 5)))
        color_index -= color_step

    return res_col_arr

def draw_and_save_graphics_many_agents(xs_arr, ys_arr, ts_arr, path_save_graphs, plot_colors, _k_elements, step_graphs=50):

    num_frames = len(xs_arr[0])

    for i in range(0+step_graphs, num_frames, step_graphs):
        plt.figure(figsize=[8,8])

        for agent in range(_k_elements):
            if(i < 50):
                plt.plot(xs_arr[agent][:i], ys_arr[agent][:i], color=plot_colors[agent])
                plt.scatter(xs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])
            else:
                plt.plot(xs_arr[agent][i-100:i], ys_arr[agent][i-100:i], color=plot_colors[agent])
                plt.scatter(xs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.suptitle(str(i) + 'time: ' + str(round(ts_arr[i], 5)))

        plt.savefig(path_save_graphs + '/t_' + str(i) + '.png')
        plt.close()

    return 0

def main():
    print('Start time:', hms_now())

    global w
    w = generate_w_arr(k_elements, _range=[0.9, 1.1])

    start_time = time.time()

    rand_IC = m.generate_random_IC_ressler(2., 2., 1.5, k_elements)
    sol = solve_ivp(func_rossler_3_dim, [0, t_max], rand_IC, rtol=1e-11, atol=1e-11)

    xs, ys, zs = [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t

    
    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_time, 'time:', hms_now())

    plot_colors = make_colors(k_elements)
    
    gs_kw = dict(width_ratios=[1.5, 1], height_ratios=[1,1,1])
    fig, axd = plt.subplot_mosaic([['xt', 'yx',],
                                   ['yt', 'xz',],
                                   ['zt', 'yz',]],
                                gridspec_kw=gs_kw, figsize=(12, 8),
                                layout="constrained")
    for ax_n in axd:
        axd[ax_n].grid()
        axd[ax_n].set_xlabel(ax_n[1])
        axd[ax_n].set_ylabel(ax_n[0])
    fig.suptitle('Сетка мобильных агентов')

    for agent in range(k_elements):
        axd['xt'].plot(ts, xs[agent], alpha=0.3, color=plot_colors[agent])
        axd['yt'].plot(ts, ys[agent], alpha=0.3, color=plot_colors[agent])
        axd['zt'].plot(ts, zs[agent], alpha=0.3, color=plot_colors[agent])
        
        axd['yx'].plot(xs[agent], ys[agent], alpha=0.3, color=plot_colors[agent])
        axd['xz'].plot(xs[agent], zs[agent], alpha=0.3, color=plot_colors[agent])
        axd['yz'].plot(zs[agent], ys[agent], alpha=0.3, color=plot_colors[agent])

    # plt.show()

    fig_last, ax_last = plt.subplots(figsize=[10, 6])
    for agent in range(k_elements):
        ax_last.plot(xs[agent][-50:], ys[agent][-50:], color=plot_colors[agent])
        ax_last.scatter(xs[agent][-1], ys[agent][-1], color=plot_colors[agent])
    ax_last.grid()
    # plt.show()

    path_save, path_save_graphs = save_data([xs, ys, zs, ts], rand_IC, w, [fig, fig_last], ['fig_graphs', 'fig_last_state'])

    # draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, 100)
    # Анимация y(x)
    frames, fig_gif = make_frames_grid_agents(xs, ys, plot_colors, frames_step=20)
    interval = 40
    blit = True
    repeat = False
    animation = ArtistAnimation(
                fig_gif,
                frames,
                interval=interval,
                blit=blit,
                repeat=repeat)
    animation_name = path_save + '/grid_agents'
    animation.save(animation_name + '.gif', writer='pillow')


    # # Анимация большая
    # frames, frames_3d, fig, fig_3d = m.make_frames(xs, ys, zs, ts, 'Grid 4x5 agents', _k_elements = k_elements)
    # # Задержка между кадрами в мс
    # interval = 50
    # # Использовать ли буферизацию для устранения мерцания
    # blit = True
    # # Будет ли анимация циклической
    # repeat = False

    # animation = ArtistAnimation(
    #             fig,
    #             frames,
    #             interval=interval,
    #             blit=blit,
    #             repeat=repeat)

    # animation_name = path_save + '/grid_agents'
    # animation.save(animation_name + '.gif', writer='pillow')
    # animation_3d = ArtistAnimation(
    #             fig_3d,
    #             frames_3d,
    #             interval=interval,
    #             blit=blit,
    #             repeat=repeat)
    
    # animation_3d.save(animation_name + '_3d.gif', writer='pillow')
    
    print('Other time', time.time() - time_after_integrate, 'time:', hms_now())

if __name__ == '__main__':
    main()