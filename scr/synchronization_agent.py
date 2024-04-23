from default_model import func_rossler_3_dim, generate_w_arr
from scipy.integrate import solve_ivp
from config import settings as s
from matplotlib import pyplot as plt
import memory_worker as mem
import numpy as np
import time

t_max = s.t_max
k_elements = s.k_col * s.k_str
k = s.k
b = s.b
c = s.c

def mean(arr):
    arr.mean()

def find_frequency(xs, ys, zs, ts, w_arr, a):
    size = len(ts)

    phi = [[] for i in range(k_elements)]
    omega = [[] for i in range(k_elements)]

    for t in range(size):
        for agent in range(k_elements):
            x = xs[agent][t]
            y = ys[agent][t]
            z = zs[agent][t]
            w = w_arr[agent]
            phi_i = np.arctan(( w * x + a * y) / (- w * y - z))
            phi[agent].append(phi_i)
            
            # omega_i_dyddx = -w**3*x**2 - 2*w**2*a*x*y - b*w*x - z*w*x*(x-c) \
            #     - w**2*a*x*y - w*a**2*y**2 - b*a*y - a*y*z*(x-c)
            # omega_i_ddydx = -w**3*y**2 - w**2*y*z + a*w**2*x*y + a**2*w*y**2 \
            #     - w**2*y*z - w*z**2 + a*w*x*z + a**2*y*z
            omega_i_dyddx = - (w*x + a*y) * (-w**2*x-w*a*y-b-z*(x-c))
            omega_i_ddydx = (-w*y-z) * (-w**2*y-w*z+a*w*x+a**2*y)

            omega_i_zn = (w*y+z)**2 + (w*x + a*y)**2
            omega[agent].append((omega_i_ddydx + omega_i_dyddx)/omega_i_zn)
    
    omega_mean = [np.mean(agent) for agent in omega]

    print('omega mean: ', omega_mean)

    for agent in range(k_elements):
        plt.plot(ts, omega[agent])
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('omega')
    plt.show()

    for agent in range(k_elements):
        plt.plot(ts, phi[agent])
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('phi')
    plt.show()



def find_dist_between_agent(xs, ys, zs, ts):
    dists = [[], []]
    size = len(ts)
    for t in range(size):
        sum = 0
        for agent in range(1, k_elements):
            sum += max((xs[agent][t] - xs[0][t])**2 + (ys[agent][t] - ys[0][t])**2)
        dists[0].append(sum)
        dists[1].append()
    
    plt.plot(ts, dists)
    plt.grid()
    plt.show()
    return 0

def solo_experiment_depend_a(a, w_arr, IC):
    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now())
    sol = solve_ivp(func_rossler_3_dim, [0, t_max], IC, args=(w_arr, a), 
                    rtol=1e-11, atol=1e-11, method=s.method)
    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_solve_time, 'time:', mem.hms_now())

    xs, ys, zs = [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t

    plot_colors = mem.make_colors(k_elements)
    
    # find_dist_between_agent(xs, ys, zs, ts, w_arr, a)

    # Order params
    find_frequency(xs, ys, zs, ts, w_arr, a)

    # Plot last state
    fig_last, ax_last = plt.subplots(figsize=[10, 6])
    for agent in range(k_elements):
        ax_last.plot(xs[agent][-50:], ys[agent][-50:], color=plot_colors[agent])
        ax_last.scatter(xs[agent][-1], ys[agent][-1], color=plot_colors[agent])
    ax_last.grid()
    ax_last.set_xlabel('x')
    ax_last.set_ylabel('y')


    return fig_last, [xs, ys, zs, ts]

def experiments_series_depend_a(a_arr, n_exps_in_one_cycle = 100):
    return 0

w_arr = generate_w_arr(k_elements)
IC = mem.generate_random_IC_ressler(10., 10., 1, k_elements)
fig = solo_experiment_depend_a(s.a, w_arr, IC)
plt.show()