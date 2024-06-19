from default_model import func_rossler_3_dim, synchronization_event, func_rossler_2_dim
from scipy.integrate import solve_ivp
import settings as s
from matplotlib import pyplot as plt
import memory_worker as mem
import numpy as np
import time
import joblib

t_max = s.t_max
if s.exps_type == 'grid':
    k_elements = s.k_str * s.k_col
else:
    k_elements = s.k_elements
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
            
            if max(dist) < 1.3 * s.radius:
                synchronization_time = ts[step*(t + 1)]
                break

    if synchronization_time == -10:
        synchronization_time = 530

    return synchronization_time, fig

def solo_experiment_depend_a_tau_p(a, w_arr, IC_arr, index = 0, isSolo = False, tau = s.tau, path_save = './data/temp/'):
    IC = IC_arr[index]
    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now())
    sol = solve_ivp(func_rossler_3_dim, [0, t_max], IC, args=(w_arr, a, tau), 
                    rtol=s.toch[0], atol=s.toch[1], method=s.method, events=synchronization_event)
    
    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_solve_time, 'time:', mem.hms_now())

    xs, ys, zs = [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t
    print(ts[-1])

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

    omega_fig.savefig(path_save + '/fig_last_state' + str(index) +'.png')
    plt.close(omega_fig)
    fig_last.savefig(path_save + '/fig_omega' + str(index) +'.png')
    plt.close(fig_last)

    figs = {'fig_last_state': fig_last,
            'fig_omega': omega_fig}
    return synchronization_time, figs, [[xs, ys, zs, ts]]

def experiments_series_depend_a_tau_parallel(a, n_exps_in_one_cycle = 100, 
                                             IC_fname = 'series_IC_500.txt', 
                                             tau = 1):
    start_time = time.time()
    IC_arr = mem.read_series_IC(s.temporary_path + IC_fname)
    # w_arr = mem.generate_w_arr(k_elements)
    w_arr = [0.957, 0.942, 0.939, 0.972, 1.024, 1.008, 1.059, 1.045, 0.976, 0.955, 1.058, 0.987, 
            1.044, 1.057, 0.968, 0.954, 0.976, 0.97, 1.042, 0.974, 0.985, 1.037, 0.992, 0.997, 0.952]

    global t_max
    if tau >= 5 and (a == 0.22 or a == 0.28):
        s.t_max = 100
        t_max = 100
        if tau >= 10:
            s.t_max = 50
            t_max = 50
    else:
        t_max = 200
        s.t_max = 200
    
    dir, figs_dir, times_dir = mem.make_dir_for_series_experiments(w_arr, a, n_exps_in_one_cycle, IC_fname, {'a': a, "tau": tau})

    times_of_sync = []
    
    n_streams = s.n_streams
    for exp in range(0, n_exps_in_one_cycle, n_streams):

        if exp < 2:
            continue

        print(f'Experiments {exp}-{exp+n_streams}. ')

        existance_next_steps = []
        if exp + n_streams > n_exps_in_one_cycle:
            existance_next_steps = range(exp, n_exps_in_one_cycle)
        else:
            existance_next_steps = range(exp, exp + n_streams)
        existance = joblib.Parallel(n_jobs=n_streams)(joblib.delayed(solo_experiment_depend_a_tau_p)(a, w_arr, IC_arr, exp_num, False, tau, figs_dir) for exp_num in existance_next_steps)

        for ex_num, ex in enumerate(existance):
            time_of_sync = ex[0]
            figs = ex[1]
        
            times_of_sync.append(time_of_sync)

            # for figname, fig in figs.items():
            #     fig.savefig(figs_dir + f'/{figname}_{exp}.png')
            # plt.close('all')

            # Запись итоговых времен (times_of_sync) в файл times.txt
            with open(times_dir, 'a') as f:
                print(f'{exp+ex_num} {times_of_sync[-1]}',  file=f)
    
    plt.figure()
    h = np.append(np.arange(0, 210, 10), 250)
    plt.hist(times_of_sync, h, edgecolor='darkblue')
    plt.xlim(-10, 230)
    plt.xlabel('Время синхронизации')
    plt.ylabel('Число синхронизаций')
    plt.savefig(dir + '/times_hist.png')

    print('Final time: ', time.time() - start_time, end='')
    if tau != 1:
        print('', 'tau:', tau)
    else:
        print('')

def solo_experiment_depend_a_tau_p_2dim(a, w_arr, IC_arr, index = 0, isSolo = False, tau = s.tau, path_save = './data/temp/'):
    # IC
    IC = IC_arr[index]

    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now())


    sol = solve_ivp(func_rossler_2_dim, [0, t_max], IC, args=(w_arr, a, tau), 
                    rtol=s.toch[0], atol=s.toch[1], method=s.method)
    
    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_solve_time, 'time:', mem.hms_now())

    # Sort integration data 
    xs, ys, zs = [], [], []
    for i in range(k_elements):
        xs.append(sol.y[i*k])
        ys.append(sol.y[i*k+1])
        zs.append(sol.y[i*k+2])
    ts = sol.t

    plot_colors = mem.make_colors(k_elements)
    
    # Plot last state
    fig_last, ax_last = plt.subplots(figsize=[20, 15])
    for agent in range(k_elements):
        ax_last.plot(xs[agent][-50:], ys[agent][-50:], color=plot_colors[agent])
        ax_last.scatter(xs[agent][-1], ys[agent][-1], color=plot_colors[agent])
    ax_last.grid()
    ax_last.set_xlabel('x')
    ax_last.set_ylabel('y')

    # Сохранить все данные интегрирования, если соло эксперимент
    if isSolo:
        path_save, path_save_graphs = mem.save_data([xs, ys, zs, ts], IC, w_arr, [fig_last], ['fig_last_state'], k_elements=k_elements)
        plt.close()
        mem.draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, 100)

    # Функция, которая считает время синхронизации
    synchronization_time, omega_fig = find_synchronization_time(xs, ys, zs, ts, w_arr, a)

    # Сохраняем графики
    omega_fig.savefig(path_save + '/fig_omega' + str(index) +'.png')
    plt.close(omega_fig)
    fig_last.savefig(path_save + '/fig_last_state' + str(index) +'.png')
    plt.close(fig_last)

    return synchronization_time, [[xs, ys, zs, ts]]

def exp_series_dep_a_tau_p_2dim(a, n_exps_in_one_cycle = 100, 
                                             IC_fname = 'series_IC_1000_10.txt', 
                                             tau = 1):
    
    start_time = time.time()

    # take IC and w
    IC_arr, w_arr = mem.read_series_IC(s.temporary_path + IC_fname)

    # Если tau большое, для экономии времени уменьшаем время интегрирования
    global t_max
    if tau >= 5 and (a == 0.22 or a == 0.28):
        s.t_max = 400
        t_max = 400
        if tau >= 10:
            s.t_max = 350
            t_max = 350
    else:
        t_max = 500
        s.t_max = 500
    
    dir, figs_dir, times_dir = mem.make_dir_for_series_experiments(w_arr, a, 
                                    n_exps_in_one_cycle, IC_fname, {'a': a, "tau": tau})

    times_of_sync = []          # Итоговый массив времен, постоянно дозаписывается
    n_streams = s.n_streams     # Число потоков для параллельной реализации
    for exp in range(0, n_exps_in_one_cycle, n_streams):
        print(f'Experiments {exp}-{exp+n_streams}. ')

        # Подбираем номера итераций для всех потоков для нового цикла
        existance_next_steps = []
        if exp + n_streams > n_exps_in_one_cycle:
            existance_next_steps = range(exp, n_exps_in_one_cycle)
        else:
            existance_next_steps = range(exp, exp + n_streams)

        # Запускаем параллельно серию одиночных экспериментов
        existance = joblib.Parallel(n_jobs=s.n_streams)(joblib.delayed(solo_experiment_depend_a_tau_p_2dim)
                        (a, w_arr, IC_arr, exp_num, False, tau, figs_dir) for exp_num in existance_next_steps)

        # Достаем данные, полученные из каждого потока
        for ex_num, ex in enumerate(existance):
            time_of_sync = ex[0]
        
            times_of_sync.append(time_of_sync)

            # Запись итоговых времен (times_of_sync) в файл times.txt
            with open(times_dir, 'a') as f:
                print(f'{exp+ex_num} {times_of_sync[-1]}',  file=f)
    
    # Рисуем итоговую гистограмму
    plt.figure(figsize=[25, 15])
    h = np.append(np.arange(0, 510, 20), 550)
    plt.hist(times_of_sync, h, edgecolor='darkblue')
    plt.xlim(-10, 530)
    plt.xlabel('Время синхронизации')
    plt.ylabel('Число синхронизаций')
    plt.savefig(dir + '/times_hist.png')

    # Конец
    print('Final time: ', time.time() - start_time, end='')
    if tau != 1:
        print('', 'tau:', tau)
    else:
        print('')

# path = mem.generate_and_write_series_IC((5., 5., 1.), n_exps=1000, k_elements=k_elements)
# Solo experiment
IC_arr, w_arr = mem.read_series_IC(s.temporary_path + 'series_IC_1000_10.txt')
for i in range(1, 100):
    synchronization_time, _ = solo_experiment_depend_a_tau_p_2dim(s.a, w_arr, IC_arr, index=i, isSolo=True)
    print(i, 'Sync time:', synchronization_time)

# Parallel series
# tau_arr = [0.1, 0.5, 1, 2, 5, 10]
# IC_file_name = 'series_IC_1000_10.txt'
# s.a = 0.22
# for tau in tau_arr:
#     exp_series_dep_a_tau_p_2dim(0.22, 1000, IC_file_name, tau=tau)
