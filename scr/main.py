from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import time
import settings as s
from datetime import datetime
from matplotlib.animation import ArtistAnimation
from model import generate_w_arr, func_rossler_3_dim
import memory_worker as mem


w = s.w                     #
a = s.a                     # Параметры
b = s.b                     # системы
c = s.c                     #

k_elements = s.k_elements   # Число агентов
k = 3                       # Число уравнений для одного агента (всегда 3)

radius = s.radius           # Радиус связи
T = s.T                     # Сила связи


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


def func_rossler_3_dim(t, r):
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

def main(IC_range = [5, 5, 0]):
    start_time = time.time()
    print('Start time:', datetime.now().time())

    global w
    w = generate_w_arr(k_elements, _range=[0.93, 1.07])

    rand_IC = mem.generate_random_IC_ressler(IC_range[0], IC_range[1], IC_range[2])
    sol = solve_ivp(func_rossler_3_dim, [0, 150], rand_IC, rtol=1e-11, atol=1e-11)

    xs, ys, zs = [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t

    
    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_time, 'time:', datetime.now().time())

    # animation
    frames, frames_3d, fig, fig_3d = mem.make_frames(xs, ys, zs, ts, '')

    time_after_make_frames = time.time()
    print('Make frames time:', time.time() - time_after_integrate, 'time:', datetime.now().time())

    # Задержка между кадрами в мс
    interval = 50
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

    animation_name = './data/gif/stop_agents_1'
    animation.save(animation_name + '.gif', writer='pillow')
    animation_3d = ArtistAnimation(
                fig_3d,
                frames_3d,
                interval=interval,
                blit=blit,
                repeat=repeat)
    
    animation_3d.save(animation_name + '_3d.gif', writer='pillow')

    plt.show()
    print('anim generate time:', time.time() - time_after_make_frames, 'time:', datetime.now().time())
    # end animation

    plot_colors = mem.make_colors(k_elements)
    path_save, path_save_graphs = mem.save_data([xs, ys, zs, ts], rand_IC, w, k_elements=k_elements)

    mem.draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, t_step=0.1)

    if s.look_at_infinity:
        for agent in range(k_elements):
            print('IC')
            print(rand_IC[agent * k], rand_IC[agent * k + 1], rand_IC[agent * k + 2])
            print('Last state:')
            print(xs[agent][-1], ys[agent][-1], zs[agent][-1])
            if xs[agent][-1] > 1000 or  ys[agent][-1] > 1000 or zs[agent][-1] > 1000:
                print('Some agent run on infinity')

    print('save time:', time.time() - time_after_integrate, 'time:', datetime.now().time())

if __name__ == '__main__':
    main()
    main()

    # 15 8 14 14 8 13