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

def find_synchronization_time(xs, ys, zs, ts, w_arr, a):
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
            
            omega_i_dyddx = - (w*x + a*y) * (-w**2*x-w*a*y-b-z*(x-c))
            omega_i_ddydx = (-w*y-z) * (-w**2*y-w*z+a*w*x+a**2*y)

            omega_i_zn = (w*y+z)**2 + (w*x + a*y)**2
            omega[agent].append((omega_i_ddydx + omega_i_dyddx)/omega_i_zn)
    
    step = 300
    omega_new = [[] for i in range(k_elements)]
    for agent in range(k_elements):
        for t in range(step, size, step):
            omega_new[agent].append(np.mean(omega[agent][t-step:t]))

    fig = plt.figure(figsize=(60, 50))
    for agent in range(k_elements):
        plt.plot(ts[step::step], omega_new[agent])
    plt.grid()
    plt.xlabel('t', fontsize=20)
    plt.ylabel('\u03A9', fontsize=20)

    max_omega_diff = 0.25
    synchronization_time = -10
    for t in range(len(omega_new[0])):
        omega_t = [omega_agent[t] for omega_agent in omega_new]
        if max(omega_t) - min(omega_t) < max_omega_diff:
            # Проверить, что расстояние между всеми агентами меньше 1.5r
            dist = []
            for agent in range(1, k_elements):
                dist.append(np.sqrt((xs[agent][step*t] - xs[0][step*t])**2+ \
                        (ys[agent][step*t] - ys[0][step*t])**2 + \
                        (zs[agent][step*t] - zs[0][step*t])**2))
            
            if max(dist) < 1.5 * s.radius:
                synchronization_time = ts[step*(t + 1)]
                break



    return synchronization_time, fig

def solo_experiment_depend_a(a, w_arr, IC, isSolo = False):
    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now())
    print('')
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
    
    # Plot last state
    fig_last, ax_last = plt.subplots(figsize=[10, 6])
    for agent in range(k_elements):
        ax_last.plot(xs[agent][-50:], ys[agent][-50:], color=plot_colors[agent])
        ax_last.scatter(xs[agent][-1], ys[agent][-1], color=plot_colors[agent])
    ax_last.grid()
    ax_last.set_xlabel('x')
    ax_last.set_ylabel('y')

    if isSolo:
        path_save, path_save_graphs = mem.save_data([xs, ys, zs, ts], IC, w_arr, [fig_last], ['fig_last_state'])
        plt.close()
        mem.draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, 100)

        # Order params
        # find_frequency(xs, ys, zs, ts, w_arr, a)

    # Функция, которая считает время синхронизации
    synchronization_time, omega_fig = find_synchronization_time(xs, ys, zs, ts, w_arr, a)
    
    if isSolo:
        print(synchronization_time)
        omega_fig.savefig(path_save + '/omega.png')

    figs = {'fig_last_state': fig_last,
            'fig_omega': omega_fig}
    return synchronization_time, figs, [[xs, ys, zs, ts]]

def experiments_series_depend_a(a, n_exps_in_one_cycle = 100, IC_fname = 'series_IC_500.txt'):
    start_time = time.time()
    IC_arr = mem.read_series_IC(s.temporary_path + IC_fname)
    # w_arr = generate_w_arr(k_elements)
    w_arr = [0.957, 0.942, 0.939, 0.972, 1.024, 1.008, 1.059, 1.045, 0.976, 0.955, 1.058, 0.987, 
            1.044, 1.057, 0.968, 0.954, 0.976, 0.97, 1.042, 0.974, 0.985, 1.037, 0.992, 0.997, 0.952]

    dir, figs_dir = mem.make_dir_for_series_experiments(w_arr, a, n_exps_in_one_cycle, IC_fname)

    times_of_sync = []
    for exp in range(n_exps_in_one_cycle):
        print(f'Exp {exp + 1}. ', end='')
        time_of_sync, figs, _ = solo_experiment_depend_a(a, w_arr, IC_arr[exp])
        times_of_sync.append(time_of_sync)

        for figname, fig in figs.items():
            fig.savefig(figs_dir + f'/{figname}_{exp}.png')
        plt.close()

    # Запись итоговых времен (times_of_sync) в файл times.txt
    with open(dir + '/times.txt', 'w') as f:
        for i in range(n_exps_in_one_cycle):
            print(f'{i+1} {times_of_sync[i]}',  file=f)
    
    plt.figure()
    plt.hist(times_of_sync, int(s.t_max/10.))
    plt.grid()
    plt.xlabel('Время синхронизации')
    plt.ylabel('Число синхронизаций')
    plt.savefig(dir + '/times_hist.png')

    print('Final time: ', time.time() - start_time)

# path = mem.generate_and_write_series_IC((5., 5., 1.), n_exps=500, k_elements=k_elements)
# w_arr = generate_w_arr(k_elements)
# IC_arr = mem.read_series_IC(s.temporary_path + 'series_IC_500.txt')
# fig = solo_experiment_depend_a(s.a, w_arr, IC_arr[0], isSolo=True)
# plt.show()


# path = mem.generate_and_write_series_IC((5., 5., 1.), n_exps=500, k_elements=k_elements)
IC_file_name = 'series_IC_500.txt'
experiments_series_depend_a(0.16, 100, IC_file_name)
s.a = 0.28
experiments_series_depend_a(0.28, 100, IC_file_name)
