from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import time
from settings.config import settings as s
from random import uniform
from datetime import datetime

w = [0.98, 1.0, 0.94, 1.07, 1.02]
a = 0.22        # Параметры
b = 0.1         # системы
c = 8.5         #

k_elements = 5  # Число агентов
k = 3           # Число уравнений для одного агента (всегда 3)

radius = 3      # Радиус связи
T = 0.2         # Сила связи

plot_colors = [
    "blue",
    "orange",
    "green",
    "red",
    "indigo",
    "m",
    "purple",
    "gray",
    "olive",
    "pink",
    "black",
    "salmon",
    "tomato",
    "navy",
    "lime",
    "orchid",
]
# Функции синхронизации

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r):
    return 0


# Функция связи по y. Последовательное движение цепочки агентов
def func_connect_y(index, r):
    summ = 0
    for i in range(k_elements):
        summ += d(T, radius, r[i*k], r[index*k], r[i*k+1], r[index*k+1]) * (r[i*k + 1] - r[index*k + 1])
    return summ


def func_connect_y_3dim(index, r):
    summ = 0
    for i in range(k_elements):
        summ += d_3dim(T, radius, r[i*k], r[index*k], r[i*k+1], r[index*k+1], r[i*k+2], r[index*k+2]) \
                * (r[i*k + 1] - r[index*k + 1])
    return summ


# Функция связи по x. Параллельное движение цепочки агентов
def func_connect_x(index, r):
    summ1, summ2 = 0, 0
    for j in range(k_elements):   
        if j != index:
            summ1 += d(T, radius, r[j*k], r[index*k], r[j*k+1], r[index*k+1]) * (r[j*k] - r[index*k])
            summ2 += d(T, radius, r[j*k], r[index*k], r[j*k+1], r[index*k+1]) / (r[index*k] - r[j*k])
            
    return summ1 + summ2


def func_connect_x_3dim(index, r):
    summ1, summ2 = 0, 0
    for j in range(k_elements):
        if j != index:
            summ1 += d_3dim(T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
                     * (r[j * k] - r[index * k])
            summ2 += d_3dim(T, radius, r[j * k], r[index * k], r[j * k + 1], r[index * k + 1], r[j*k + 2], r[index*k+2]) \
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
def func_dx(index, r, connect_f=default_f):
    return - w[index] * r[index*k + 1] - r[index*k + 2] + connect_f(index, r)


# По y.
def func_dy(index, r, connect_f=default_f):
    return w[index] * r[index*k] + a * r[index*k + 1] + connect_f(index, r)


# По z
def func_dz(index, r, connect_f=default_f):
    return b + r[index*k + 2] * (r[index*k] - c) + connect_f(index, r)


def func_rossler_2_dim(t, r):
    res_arr = []

    for i in range(k_elements):
        # x_i = r[i*k]
        # y_i = r[i*k + 1]
        # z_i = r[i*k + 2]

        dx = func_dx(i, r, func_connect_x_3dim)
        dy = func_dy(i, r, func_connect_y_3dim)
        dz = func_dz(i, r)

        res_arr.append(dx)
        res_arr.append(dy)
        res_arr.append(dz)

    return res_arr


def generate_random_IC_ressler(x_max, y_max, z_max, k_elems=k_elements, x_min='',  y_min='',  z_min=''):
    if x_min == '':
        x_min = - x_max
    if y_min == '':
        y_min = - y_max
    if z_min == '':
        z_min = - z_max
    
    res_IC_arr = []
    for i in range(k_elems):
        res_IC_arr.append(uniform(x_min, x_max))
        res_IC_arr.append(uniform(y_min, y_max))
        res_IC_arr.append(uniform(z_min, z_max))

    return res_IC_arr


def main():
    start_time = time.time()

    rand_IC = generate_random_IC_ressler(3, 3, 3)
    sol = solve_ivp(func_rossler_2_dim, [0, 100], rand_IC, rtol=1e-11, atol=1e-11)

    xs, ys, zs = [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t


    from matplotlib.animation import ArtistAnimation

    def make_frames(xs_arr, ys_arr, zs_arr, ts_arr, axd, ax_3D):
        # Кол-во кадров
        num_frames = len(xs_arr[0])
        
        frames = []
        frames_3d = []
        # Совершаем проход и вырисовываем каждую пятую точку как новый кадр
        for i in range(0, num_frames, 10):
            frame = []
            frame_3d = []

            # Цикл по каждому агенту (элементу)
            for agent in range(k_elements):

                
                # Очень старая линия почти не видима, чтобы не мешалась (только на графиках xt, xyz)
                if i > 2000:
                    # xy
                    xy_line_very_transperent, = axd['yx'].plot(xs_arr[agent][:i-2000], ys_arr[agent][:i-2000], color=plot_colors[agent], alpha=0.15)
                    xy_line_transperent, = axd['yx'].plot(xs_arr[agent][i-2001:i-500], ys_arr[agent][i-2001:i-500], color=plot_colors[agent], alpha=0.4)
                    xy_line, = axd['yx'].plot(xs_arr[agent][i-501:i], ys_arr[agent][i-501:i], color=plot_colors[agent])
                    frame.append(xy_line_very_transperent)
                    frame.append(xy_line_transperent)
                    frame.append(xy_line)

                    # xz
                    xz_line_very_transperent, = axd['xz'].plot(zs_arr[agent][:i-2000], xs_arr[agent][:i-2000], color=plot_colors[agent], alpha=0.15)
                    xz_line_transperent, = axd['xz'].plot(zs_arr[agent][i-2001:i-500], xs_arr[agent][i-2001:i-500], color=plot_colors[agent], alpha=0.4)
                    xz_line, = axd['xz'].plot(zs_arr[agent][i-501:i], xs_arr[agent][i-501:i], color=plot_colors[agent])
                    frame.append(xz_line_very_transperent)
                    frame.append(xz_line_transperent)
                    frame.append(xz_line)

                    # yz
                    yz_line_very_transperent, = axd['yz'].plot(zs_arr[agent][:i-2000], ys_arr[agent][:i-2000], color=plot_colors[agent], alpha=0.15)
                    yz_line_transperent, = axd['yz'].plot(zs_arr[agent][i-2001:i-500], ys_arr[agent][i-2001:i-500], color=plot_colors[agent], alpha=0.4)
                    yz_line, = axd['yz'].plot(zs_arr[agent][i-501:i], ys_arr[agent][i-501:i], color=plot_colors[agent])
                    frame.append(yz_line_very_transperent)
                    frame.append(yz_line_transperent)
                    frame.append(yz_line)

                    xyz_line_very_transperent, = ax_3D.plot3D(xs_arr[agent][:i-2000], ys_arr[agent][:i-2000], zs_arr[agent][:i-2000], color=plot_colors[agent], alpha=0.15)
                    xyz_line_transperent, = ax_3D.plot3D(xs_arr[agent][i-2001:i-500], ys_arr[agent][i-2001:i-500], zs_arr[agent][i-2001:i-500], color=plot_colors[agent], alpha=0.4)
                    xyz_line, = ax_3D.plot3D(xs_arr[agent][i-501:i], ys_arr[agent][i-501:i], zs_arr[agent][i-501:i], color=plot_colors[agent])
                    frame_3d.append(xyz_line_very_transperent)
                    frame_3d.append(xyz_line_transperent)
                    frame_3d.append(xyz_line)

                elif i > 500:
                    #xy
                    xy_line_transperent, = axd['yx'].plot(xs_arr[agent][:i-500], ys_arr[agent][:i-500], color=plot_colors[agent], alpha=0.4)
                    xy_line, = axd['yx'].plot(xs_arr[agent][i-501:i], ys_arr[agent][i-501:i], color=plot_colors[agent])
                    frame.append(xy_line_transperent)
                    frame.append(xy_line)
                    
                    #xz
                    xz_line_transperent, = axd['xz'].plot(zs_arr[agent][:i-500], xs_arr[agent][:i-500], color=plot_colors[agent], alpha=0.4)
                    xz_line, = axd['xz'].plot(zs_arr[agent][i-501:i], xs_arr[agent][i-501:i], color=plot_colors[agent])
                    frame.append(xz_line_transperent)
                    frame.append(xz_line)
                    
                    #yz
                    yz_line_transperent, = axd['yz'].plot(zs_arr[agent][:i-500], ys_arr[agent][:i-500], color=plot_colors[agent], alpha=0.4)
                    yz_line, = axd['yz'].plot(zs_arr[agent][i-501:i], ys_arr[agent][i-501:i], color=plot_colors[agent])
                    frame.append(yz_line_transperent)
                    frame.append(yz_line)

                    #xyz
                    xyz_line_transperent, = ax_3D.plot3D(xs_arr[agent][:i-500], ys_arr[agent][:i-500], zs_arr[agent][:i-500], color=plot_colors[agent], alpha=0.4)
                    xyz_line, = ax_3D.plot3D(xs_arr[agent][i-501:i], ys_arr[agent][i-501:i], zs_arr[agent][i-501:i], color=plot_colors[agent])
                    frame_3d.append(xyz_line_transperent)
                    frame_3d.append(xyz_line)

                else: 
                    #xy
                    xy_line, = axd['yx'].plot(xs_arr[agent][:i], ys_arr[agent][:i], color=plot_colors[agent])
                    frame.append(xy_line)
                    #xz
                    xy_line, = axd['xz'].plot(zs_arr[agent][:i], xs_arr[agent][:i], color=plot_colors[agent])
                    frame.append(xy_line)
                    #yz
                    xy_line, = axd['yz'].plot(zs_arr[agent][:i], ys_arr[agent][:i], color=plot_colors[agent])
                    frame.append(xy_line)
                    #xyz
                    xyz_line, = ax_3D.plot3D(xs_arr[agent][:i], ys_arr[agent][:i], zs_arr[agent][:i], color=plot_colors[agent])
                    frame_3d.append(xyz_line)

                # Последние точки xy, xz, yz
                xy_point = axd['yx'].scatter(xs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])
                xz_point = axd['xz'].scatter(zs_arr[agent][i], xs_arr[agent][i], color=plot_colors[agent])
                yz_point = axd['yz'].scatter(zs_arr[agent][i], ys_arr[agent][i], color=plot_colors[agent])
                frame.append(xy_point)
                frame.append(xz_point)
                frame.append(yz_point)
                #xyz
                xyz_point = ax_3D.scatter(xs_arr[agent][i], ys_arr[agent][i], zs_arr[agent][i], color=plot_colors[agent])
                frame_3d.append(xyz_point)

                # Рисуем графики x(t), y(t), z(t)
                xt_line, = axd['xt'].plot(ts_arr[:i], xs_arr[agent][:i], color=plot_colors[agent])
                yt_line, = axd['yt'].plot(ts_arr[:i], ys_arr[agent][:i], color=plot_colors[agent])
                zt_line, = axd['zt'].plot(ts_arr[:i], zs_arr[agent][:i], color=plot_colors[agent])
                frame.append(xt_line)
                frame.append(yt_line)
                frame.append(zt_line)

                # Рисуем последние точки x(t), y(t), z(t)
                xt_point = axd['xt'].scatter(ts_arr[i], xs_arr[agent][i], color=plot_colors[agent])
                yt_point = axd['yt'].scatter(ts_arr[i], ys_arr[agent][i], color=plot_colors[agent])
                zt_point = axd['zt'].scatter(ts_arr[i], zs_arr[agent][i], color=plot_colors[agent])
                frame.append(xt_point)
                frame.append(yt_point)
                frame.append(zt_point)
                
            frames.append(frame)
            frames_3d.append(frame_3d)

        return frames, frames_3d
    
    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_time, 'time:', datetime.now().time())

    # Пытаемся расположить графики
    gs_kw = dict(width_ratios=[1.5, 1], height_ratios=[1,1,1,1,1,1])
    fig, axd = plt.subplot_mosaic([['xt', 'yx',],
                                   ['xt', 'yx',],
                                   ['yt', 'xz',],
                                   ['yt', 'xz',],
                                   ['zt', 'yz',],
                                   ['zt', 'yz',]],
                                gridspec_kw=gs_kw, figsize=(12, 8),
                                layout="constrained")
    for ax_n in axd:
        axd[ax_n].grid()
        axd[ax_n].set_xlabel(ax_n[1])
        axd[ax_n].set_ylabel(ax_n[0])
    fig.suptitle('Параллельное движение агентов')

    
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')
    ax_3d.grid()


    frames, frames_3d = make_frames(xs, ys, zs, ts, axd, ax_3d)

    time_after_make_frames = time.time()
    print('Make frames time:', time.time() - time_after_integrate, 'time:', datetime.now().time())

    # Задержка между кадрами в мс
    interval = 40

    # Использовать ли буферизацию для устранения мерцания
    blit = True

    # Будет ли анимация циклической
    repeat = False

    animation = ArtistAnimation(
                fig,
                frames,
                interval=interval,
                blit=blit,
                repeat=repeat)

    animation_name = './data/gif/parallel_agents2'
    animation.save(animation_name + '.gif', writer='pillow')
    animation_3d = ArtistAnimation(
                fig_3d,
                frames_3d,
                interval=interval,
                blit=blit,
                repeat=repeat)
    
    animation_3d.save(animation_name + '_3d.gif', writer='pillow')

    #plt.show()
    print('anim generate time:', time.time() - time_after_make_frames, 'time:', datetime.now().time())

if __name__ == '__main__':
    main()