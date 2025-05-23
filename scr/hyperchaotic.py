import settings as s
import memory_worker as mem
import numpy as np
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt

def calc_phi_omega_4d(data, a, b, c, d, omega, step = 250):
    xs, ys, zs, ws, ts = data

    size = len(ts)
    phi = []
    Omega = []

    for t in range(size):
        x = xs[t]
        y = ys[t]
        z = zs[t]
        w = ws[t]

        dx = -omega*y - z
        dy = omega*x + a*y + w
        dz = b + x*z
        dw = -c*z + d*w

        if dx == 0:
            phi_i = 0
        else:
            phi_i = np.arctan(dy/dx)

        ddx = - omega*dy - dz
        ddy = omega*dx + a*dy + dw
        omega_i = (ddy*dx - dy*ddx) / (dy**2 + dx**2)
        
        phi.append(phi_i)
        Omega.append(omega_i)

    Omega_mean = []
    for t in range(step, size, step):
        Omega_mean.append(np.mean(Omega[t-step:t]))

    return phi, Omega, Omega_mean


def dist(data1, data2, k = 4):
    dist = []

    for i in range(len(data1[0])):
        summ = 0.
        for perem in range(k):
            summ += (data1[perem][i] - data2[perem][i])**2
        dist.append(np.sqrt(summ))
    
    return dist

class Coup_params:
    def __init__(self, T, k_elements, radius = 4., k = 4):
        self.T = T
        self.k_elements = k_elements
        self.radius = radius
        self.k = k

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r, T_, p):
    return 0

def func_dx(i, r, omega, k = 4, connect_f = default_f, params = None):
    return - omega[i] * r[i*k + 1] - r[i*k + 2] + connect_f(i, r, params, 'x')

def func_dy(i, r, a, omega, k = 4, connect_f = default_f, params = None):
    return omega[i] * r[i*k] + a * r[i*k + 1] + r[i*k + 3] + connect_f(i, r, params, 'y')

def func_dz(i, r, b, k = 4):
    return b + r[i*k] * r[i*k + 2]

def func_dw(i, r, c, d, k = 4):
    return - c * r[i*k + 2] + d * r[i*k + 3]


couplings_on = False
# Функция включения связи между двумя агентами
def d(_T, _radius, x_i, x_j, y_i, y_j):
    global couplings_on
    if (x_i - x_j)**2 + (y_i - y_j)**2 < _radius**2 or couplings_on:
        couplings_on = True
        return _T
    else:
        return 0
    
def f_connect_st(i, r, params, perem = 'y'):
    if perem == 'z':
        p_shift = 2
    if perem == 'x':
        p_shift = 0
    else:
        p_shift = 1

    k = params.k
    summ = 0
    for j in range(params.k_elements):
        if j != i:
            summ += d(params.T, params.radius, r[j*k], r[i*k], r[j*k+1], r[i*k+1]) * (r[j*k + p_shift] - r[i*k + p_shift])
    return summ

def func_rossler_4d_params_maker(a, b, c, d, T = 0.3, k_elements = 2, tau = 1, radius = 4., omega = [0.99, 1.01]):
    k = 4

    params = Coup_params(T, k_elements, radius, k)

    def func_rossler_4d_params(t, r):
        res_arr = []

        for i in range(k_elements):
            # x_i = r[i*k]
            # y_i = r[i*k + 1]
            # z_i = r[i*k + 2]
            # w_i = r[i*k + 3]

            # dx = - y_i - z_i
            # dy = x_i + a * y_i + w_i
            # dz = b + x_i * z_i
            # dw = -c * z_i + d * w_i

            dx = func_dx(i, r, omega)
            dy = func_dy(i, r, a, omega, k, connect_f=f_connect_st, params=params)
            dz = func_dz(i, r, b)
            dw = func_dw(i ,r, c, d)

            if max(r[i*k], r[i*k+1], r[i*k+2], r[i*k+3]) > 10000:
                print('Inf', f'T: {t}')

            res_arr.append(dx)
            res_arr.append(dy)
            res_arr.append(dz)
            res_arr.append(dw)

        return res_arr

    return func_rossler_4d_params

#################################################

def calculate_dx_dy(xs, ys, zs, ws, a):

    size = len(xs)

    dx = []
    dy = []
    for i in range(size):
        dx.append(-ys[i] - zs[i])
        dy.append(xs[i] + a * ys[i] + ws[i])

    return dx, dy

def solo_experiment_4d_rossler(T = 0.01):
    # params
    a = 0.25
    b = 3
    c = 0.5
    d = 0.05
    omega = [0.99, 1.01]
    k_elements = 2
    k = 4

    # IC
    IC = [
        -7.2, 19.6, 1.3, 30.1,
        -7.1, 19.5, 1.3, 30.,
        -10., 1., 1., 10.,
        -20., 0., 0., 15.,
        -9., 0., 1., 9.5,
        0., 0., 0., 20.,
        0., 1., 0., 20. 
        ]

    # integrate
    func_rossler_4d_params = func_rossler_4d_params_maker(a, b, c, d, k_elements=k_elements, T = T, omega=omega)

    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now())
    sol = solve_ivp(func_rossler_4d_params, [0, 5000], IC[:k*k_elements], method=s.method, rtol=s.toch[0], atol=s.toch[1])

    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_solve_time, 'time:', mem.hms_now())

    xs, ys, zs, ws = [], [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
        ws.append(sol.y[i*k+3])
    ts = sol.t

    # xs = sol.y[0]
    # ys = sol.y[1]
    # zs = sol.y[2]
    # ws = sol.y[3]
    # ts = sol.t
    # print(len(ts))

    # graphs

    # Двумерные проекции
    gs_kw = dict(width_ratios=[1, 1], height_ratios=[1,1,1])
    fig, axd = plt.subplot_mosaic([['yx', 'yz',],
                                   ['xz', 'yw',],
                                   ['xw', 'wz',]],
                                gridspec_kw=gs_kw, figsize=(12, 8),
                                layout="constrained")
    for ax_n in axd:
        axd[ax_n].grid()
        axd[ax_n].set_xlabel(ax_n[1])
        axd[ax_n].set_ylabel(ax_n[0])
    fig.suptitle('Двумерные проекции')

    plot_colors = ['r', 'b']

    for agent in range(k_elements):
        axd['yx'].plot(xs[agent], ys[agent], color=plot_colors[agent])
        axd['xz'].plot(zs[agent], xs[agent], color=plot_colors[agent])
        axd['xw'].plot(ws[agent], xs[agent], color=plot_colors[agent])
        axd['yz'].plot(zs[agent], ys[agent], color=plot_colors[agent])
        axd['yw'].plot(ws[agent], ys[agent], color=plot_colors[agent])
        axd['wz'].plot(zs[agent], ws[agent], color=plot_colors[agent])
    
    new_dir, data_dir = mem.save_data([], IC, 1, k_elements=k_elements, k=k, dir_name_suffix='4dim', save_int_data=False, T=T)
    fig.savefig(new_dir + '/2d_graphs.png')
    plt.close(fig)

    # Осциллограммы 
    gs_kw = dict(width_ratios=[1, 1], height_ratios=[1,1])
    fig, axd = plt.subplot_mosaic([['xt', 'yt',],
                                   ['zt', 'wt',]],
                                gridspec_kw=gs_kw, figsize=(12, 8),
                                layout="constrained")
    for ax_n in axd:
        axd[ax_n].grid()
        axd[ax_n].set_xlabel(ax_n[1])
        axd[ax_n].set_ylabel(ax_n[0])
    fig.suptitle('Осциллограммы')

    for agent in range(k_elements):
        axd['xt'].plot(ts, xs[agent], color=plot_colors[agent])
        axd['yt'].plot(ts, ys[agent], color=plot_colors[agent])
        axd['zt'].plot(ts, zs[agent], color=plot_colors[agent])
        axd['wt'].plot(ts, ws[agent], color=plot_colors[agent])

    fig.savefig(new_dir + '/temporary_implementations.png')
    plt.close(fig)

    if k_elements == 2:
        dist_arr = dist([xs[0], ys[0], zs[0], ws[0]], 
                        [xs[1], ys[1], zs[1], ws[1]])
        
        plt.plot(ts, dist_arr)
        plt.grid()
        plt.xlabel('t')
        plt.ylabel('Евклидово расстояние между агентами')
        plt.savefig(new_dir + '/dist_png')
        plt.close()

        # Считаем средние частоты
        step = 100
        plt.figure(figsize=[20, 10])
        for agent in range(k_elements):
            phi, Omega, Omega_mean = calc_phi_omega_4d((xs[agent], ys[agent], zs[agent], ws[agent], ts), a, b, c, d, omega[agent], step=step)
            plt.plot(ts[step::step], Omega_mean, label=f'omega {agent}')
        plt.grid()
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('Среднее Omega')
        plt.savefig(new_dir + '/Omega_t.png')

    # for agent in range(k_elements):
    #     print(agent+1, xs[agent][-1], ys[agent][-1], zs[agent][-1], ws[agent][-1])

    # dx, dy = calculate_dx_dy(xs, ys, zs, ws, a)

    # plt.plot(dx, dy)
    # plt.grid()
    # plt.xlabel("x'")
    # plt.ylabel("y'")
    # plt.title("График y'(x')")
    # plt.savefig(new_dir + '/fig_dy_dx.png')
    # plt.close()

solo_experiment_4d_rossler(T = 0.0)