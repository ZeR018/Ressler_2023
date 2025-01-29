from scipy.integrate import solve_ivp
import settings as s
from matplotlib import pyplot as plt
import memory_worker as mem
import numpy as np
import time
import joblib

# Параметры
t_max = s.t_max
if s.exps_type == 'grid':
    k_elements = s.k_str * s.k_col
else:
    k_elements = s.k_elements
k = s.k
b = s.b
c = s.c
radius = s.radius
min_radius = s.min_radius
T = s.T
tau = s.tau

# Методы численного интегрирования

# Стандартная (если связи по какой-то переменной нет)
def default_f(index, r, T_, perem = 'y', k = s.k, k_elements = k_elements, radius=radius):
    return 0

# Функции правой части
# По x
def func_dx(i, r, connect_f=default_f, _T=s.T, w_arr = w_arr, connect_f_inh = default_f):
    return - w_arr[i] * r[i*k + 1] - r[i*k + 2] + connect_f(i, r, _T, 'x', k_elements=k_elements) + connect_f_inh(i, r, _T, 'x')


# По y.
def func_dy(i, r, connect_f=default_f, _T=s.T, w_arr = w_arr, connect_f_inh = default_f):
    return w_arr[i] * r[i*k] + a * r[i*k + 1] + connect_f(i, r, _T, 'y', k_elements=k_elements) + connect_f_inh(i, r, _T, 'y')


# По z
def func_dz(i, r, connect_f=default_f, _T = s.T, connect_f_inh = default_f):
    return b + r[i*k + 2] * (r[i*k] - c) + connect_f(i, r, _T, 'z', k_elements=k_elements) + connect_f_inh(i, r, _T, 'z')

# Функция включения связи между двумя агентами
def d(_T, _radius, x_i, x_j, y_i, y_j, min_radius = 0.):
    dist = (x_i - x_j)**2 + (y_i - y_j)**2
    if dist < _radius**2:
        if min_radius != 0.:
            if dist < min_radius:
                return 0
            
        return _T
    else:
        return 0

# Функция связи по x. Параллельное движение цепочки агентов
def f_connect_x_repulsive(i, r, _T, perem = 'x'):
    summ1, summ2 = 0, 0
    for j in range(k_elements):   
        if j != i:
            summ1 += d(_T, radius, r[j*k], r[i*k], r[j*k+1], r[i*k+1]) * (r[j*k] - r[i*k])
            summ2 += d(_T, radius, r[j*k], r[i*k], r[j*k+1], r[i*k+1]) / (r[i*k] - r[j*k])
            
    return summ1 + summ2

def f_connect_st(i, r, _T, perem = 'y', k=s.k, k_elements=k_elements, radius = s.radius):
    if perem == 'z':
        p_shift = 2
    if perem == 'x':
        p_shift = 0
    else:
        p_shift = 1

    summ = 0
    for j in range(k_elements):
        if j != i:
            summ += d(_T, radius, r[j*k], r[i*k], r[j*k+1], r[i*k+1], min_radius=min_radius) * (r[j*k + p_shift] - r[i*k + p_shift])
    return summ

def f_connect_inh(i, r, _T, perem = 'y'):
    if perem == 'z':
        p_shift = 2
    if perem == 'x':
        p_shift = 0
    else:
        p_shift = 1

    summ = 0
    for j in range(k_elements):
        if j != i:
            summ += d(_T, radius, r[j*k], r[i*k], r[j*k+1], r[i*k+1]) / (r[i*k + p_shift] - r[j*k + p_shift])
    return summ

def func_rossler_2_dim(t, r, w_arr_, a_, tau_ = tau):
    print(f'\033[F\033[KCurrent integrate time: {round(t, 1)};', f'last update time: {mem.hms_now()}')
    global k_elements, w_arr, a
    w_arr = w_arr_
    a = a_
    res_arr = []

    for i in range(k_elements):
        # x_i = r[i*k]
        # y_i = r[i*k + 1]
        # z_i = r[i*k + 2]

        dx = tau_ * func_dx(i, r, default_f, T, w_arr)
        dy = tau_ * func_dy(i, r, f_connect_st, T, w_arr)
        dz = tau_ * func_dz(i, r, _T = T)

        res_arr.append(dx)
        res_arr.append(dy)
        res_arr.append(dz)

    return res_arr

def calc_phi_and_omega(xs, ys, zs, ts, w, a, step = 250):
    size = len(ts)
    
    phi = []
    omega = []

    for t in range(size):
        x = xs[t]
        y = ys[t]
        z = zs[t]
        phi_i = np.arctan(( w * x + a * y) / (- w * y - z))
        phi.append(phi_i)
        
        omega_i_dyddx = - (w*x + a*y) * (-w**2*x-w*a*y-b-z*(x-c))
        omega_i_ddydx = (-w*y-z) * (-w**2*y-w*z+a*w*x+a**2*y)

        omega_i_zn = (w*y+z)**2 + (w*x + a*y)**2
        omega.append((omega_i_ddydx + omega_i_dyddx)/omega_i_zn)

    omega_mean = []
    for t in range(step, size, step):
        omega_mean.append(np.mean(omega[t-step:t]))

    return phi, omega, omega_mean


def find_synchronization_time(xs, ys, zs, ts, w_arr, a):
    phi = []
    omega_new = []
    step = 250

    for agent in range(k_elements):
        phi_agent, omega_agent, omega_mean_agent = calc_phi_and_omega(xs[agent], ys[agent], zs[agent], 
                                                                      ts, w_arr[agent], a, step)
        phi.append(phi_agent)
        omega_new.append(omega_mean_agent)
    size = len(ts)

    fig = plt.figure(figsize=(60, 50))
    for agent in range(k_elements):
        plt.plot(ts[step::step], omega_new[agent])
    plt.grid()
    plt.xlabel('t', fontsize=20)
    plt.ylabel('\u03A9', fontsize=20)

    synchronization_time = -10
    
    for t in range(1, len(omega_new[0])):

        omega_t = [omega_agent[t] for omega_agent in omega_new]
        omega_prev_t = [omega_agent[t-1] for omega_agent in omega_new]

        diff_betveen_omega_last_and_prev = [abs(omega_t[i] - omega_prev_t[i]) for i in range(k_elements)]
        mean_diff_betveen_omega_last_and_prev = np.mean(diff_betveen_omega_last_and_prev)
        deviation_from_mean_omega = [abs(mean_diff_betveen_omega_last_and_prev - diff_betveen_omega_last_and_prev[i]) for i in range(k_elements)]

        max_div = 0.01
        if a == 0.28:
            max_div = 0.1
        if a == 0.22:
            max_div = 0.05
        if max(deviation_from_mean_omega) >= max_div:
            continue

        # Проверить, что расстояние между всеми агентами меньше 1.5r
        dist = []
        for agent in range(1, k_elements):
            dist.append(np.sqrt((xs[agent][step*t] - xs[0][step*t])**2+ \
                    (ys[agent][step*t] - ys[0][step*t])**2 + \
                    (zs[agent][step*t] - zs[0][step*t])**2))
        
        if max(dist) >= 1.5 * s.radius:
            continue

        # Проверить, что все агенты находятся в относительно близких фазах
        phi_t = [phi[agent][step*t] for agent in range(k_elements)]
        mean_phi_t = np.mean(phi_t)
        deviation_phi = [abs(phi_t[agent] - mean_phi_t) for agent in range(k_elements)]

        if max(deviation_phi) > np.pi / 8.0:
            continue
        else:
            synchronization_time = ts[step*(t-1)]
            break
    
    if synchronization_time == -10:
        synchronization_time = 530

    # Проверка на синхронность в последний момент 
    dist_last = []
    sync_last_time = True
    for agent in range(1, k_elements):
        dist_last.append(np.sqrt((xs[agent][-1] - xs[0][-1])**2+ \
                (ys[agent][-1] - ys[0][-1])**2))
    if max(dist_last) > 1.3 * s.radius:
        sync_last_time = False

    return synchronization_time, sync_last_time, fig

def solo_experiment_depend_a_tau_p_2dim(a, w_arr, IC_arr, index = 0, isSolo = False, tau = s.tau, path_save = './data/temp/', find_sync_time = True):
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

    # Функция, которая считает время синхронизации
    synchronization_time, sync_last_time, omega_fig = find_synchronization_time(xs, ys, zs, ts, w_arr, a)

    # Сохранить все данные интегрирования, если соло эксперимент
    if isSolo:
        path_save, path_save_graphs = mem.save_data([xs, ys, zs, ts], IC, w_arr, [fig_last], ['fig_last_state'], k_elements=k_elements, a=a, tau=tau)
        plt.close()
        mem.draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, t_step=0.1)

        plt.plot()

    # Сохраняем графики
    omega_fig.savefig(path_save + '/fig_omega' + str(index) +'.png')
    plt.close(omega_fig)
    fig_last.savefig(path_save + '/fig_last_state' + str(index) +'.png')
    plt.close(fig_last)

    return synchronization_time, sync_last_time, [[xs, ys, zs, ts]]

def exp_series_dep_a_tau_p_2dim(a, n_exps_in_one_cycle = 100, 
                                             IC_fname = 'series_IC_1000_10.txt', 
                                             tau = 1):
    
    start_time = time.time()

    # take IC and w
    IC_arr, w_arr = mem.read_series_IC(s.temporary_path + IC_fname)
    
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
            sync_last_time = ex[1]
        
            times_of_sync.append(time_of_sync)

            # Запись итоговых времен (times_of_sync) в файл times.txt
            with open(times_dir, 'a') as f:
                print(f'{exp+ex_num} {times_of_sync[-1]}',  file=f, end='')
                if not sync_last_time:
                    print(f'\tnsl', file = f)
                else:
                    print('', file = f)
    
    # Рисуем итоговую гистограмму
    plt.figure(figsize=[25, 15])
    h = np.append(np.arange(0, 510, 20), 550)
    plt.hist(times_of_sync, h, edgecolor='darkblue')
    plt.xlim(-10, 530)
    plt.xlabel('Время синхронизации')
    plt.ylabel('Число синхронизаций')
    plt.savefig(dir + '/times_hist.png')
