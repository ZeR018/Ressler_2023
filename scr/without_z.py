import settings as s
import memory_worker as mem
import numpy as np
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt

class Coup_params:
    def __init__(self, T, k_elements, radius = 4., k = 2):
        self.T = T
        self.k_elements = k_elements
        self.radius = radius
        self.k = k

def rossler_without_z_maker(a, w_arr, k_elements, radius = 4., T = 0.3):
    k = 2

    if type(w_arr) == float:
        w_arr = [w_arr for i in range(k_elements)]

    params = Coup_params(T, k_elements, radius, k)

    def rossler_without_z(t, r):
        res_arr = []

        for i in range(k_elements):
            dx = - w_arr[i] * r[i*k + 1]
            dy = w_arr[i] * r[i*k] + a * r[i*k + 1]

            res_arr.append(dx)
            res_arr.append(dy)
        return res_arr
    
    return rossler_without_z

def make_experiment_without_z():
    # params
    a = 0.22
    k_elements = 1
    k = 2
    T = 0.3
    w_arr = [1.]

    # IC
    IC = [
        1., 1.
          ]
    print('IC:', IC)

    # integrate
    func_rossler_without_z = rossler_without_z_maker(a, w_arr=w_arr, k_elements=k_elements, T = T)

    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now())
    sol = solve_ivp(func_rossler_without_z, [0, 100], IC, method=s.method, rtol=s.toch[0], atol=s.toch[1])

    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_solve_time, 'time:', mem.hms_now())

    xs, ys, zs, ws = [], [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
    ts = sol.t

    new_dir, data_dir = mem.save_data([], IC, w_arr, k_elements=k_elements, k=k, dir_name_suffix='2dim', save_int_data=False, T=T)

    plot_colors = ['orange', 'b']

    data_names = ['x', 'y', 'z', 'w']
    sol_data = [xs, ys]
    for i, data in enumerate(sol_data):
        plt.figure(figsize=[6, 4])
        plt.xlabel('t')
        plt.grid()
        plt.ylabel(data_names[i])
        for agent in range(k_elements):
            plt.plot(ts, data[agent], color=plot_colors[agent])
        plt.savefig(new_dir + f'/fig_{data_names[i]}_t.png')
        plt.close()

make_experiment_without_z()