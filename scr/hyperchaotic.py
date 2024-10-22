import settings as s
import memory_worker as mem
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r, T_, p):
    return 0

def func_dx(i, r, k = 4):
    return - r[i*k + 1] - r[i*k + 2]

def func_dy(i, r, a, k = 4):
    return r[i*k] + a * r[i*k + 1] + r[i*k + 3]

def func_dz(i, r, b, k = 4):
    return b + r[i*k] * r[i*k + 2]

def func_dw(i, r, c, d, k = 4):
    return - c * r[i*k + 2] + d * r[i*k + 3]

def func_rossler_4d_params_maker(a, b, c, d, T = 0.3, k_elements = 1, tau = 1):
    k = 4

    def func_rossler_4d_params(t, r):
        res_arr = []

        for i in range(k_elements):
            x_i = r[i*k]
            y_i = r[i*k + 1]
            z_i = r[i*k + 2]
            w_i = r[i*k + 3]

            dx = - y_i - z_i
            dy = x_i + a * y_i + w_i
            dz = b + x_i * z_i
            dw = -c * z_i + d * w_i

            # dx = func_dx(i, r)
            # dy = func_dy(i, r, a)
            # dz = func_dz(i, r, b)
            # dw = func_dw(i ,r, c, d)

            res_arr.append(dx)
            res_arr.append(dy)
            res_arr.append(dz)
            res_arr.append(dw)

        return res_arr

    # def func_rossler_4d_params(t, r):
    #     x, y, z, w = r

    #     dx = 

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

def solo_experiment_4d_rossler():
    # params
    a = 0.25
    b = 3
    c = 0.5
    d = 0.05

    # IC
    IC = (-10., 1., 1., 10)
    print('IC:', IC)

    # integrate
    func_rossler_4d_params = func_rossler_4d_params_maker(a, b, c, d)

    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now())
    sol = solve_ivp(func_rossler_4d_params, [0, 100], IC, method=s.method, rtol=s.toch[0], atol=s.toch[1])

    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_solve_time, 'time:', mem.hms_now())

    xs = sol.y[0]
    ys = sol.y[1]
    zs = sol.y[2]
    ws = sol.y[3]
    ts = sol.t
    print(len(ts))

    # graphs

    # Пытаемся расположить графики
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

    axd['yx'].plot(xs, ys)
    axd['xz'].plot(zs, xs)
    axd['xw'].plot(ws, xs)

    axd['yz'].plot(zs, ys)
    axd['yw'].plot(ws, ys)
    axd['wz'].plot(zs, ws)

    new_dir, data_dir = mem.save_data([], IC, 1, k_elements=1, k=4, dir_name_suffix='4dim', save_int_data=False)
    fig.savefig(new_dir + '/2d_graphs.png')
    plt.close(fig)

    data_names = ['x', 'y', 'z', 'w']
    for i, data in enumerate(sol.y):
        plt.figure(figsize=[6, 4])
        plt.grid()
        plt.plot(ts, data)
        plt.xlabel('t')
        plt.ylabel(data_names[i])
        plt.savefig(new_dir + f'/fig_{data_names[i]}_t.png')
        plt.close()

    dx, dy = calculate_dx_dy(xs, ys, zs, ws, a)

    plt.plot(dx, dy)
    plt.grid()
    plt.xlabel("x'")
    plt.ylabel("y'")
    plt.title("График y'(x')")
    plt.savefig(new_dir + '/fig_dy_dx.png')
    plt.close()


solo_experiment_4d_rossler()