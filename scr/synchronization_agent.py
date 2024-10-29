from default_model import func_rossler_3_dim, synchronization_event, func_rossler_2_dim, func_rossler_2_dim_params_maker
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

# Подсчет Фи и Омеги вынесен в отдельную функцию
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

        # Проверить, что изменение угловых скоростей всех агентов происходит симметрично 
        # (с шагом угловая скорость всех агентов изменяется примерно одинаково)
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

    # Делаем часть графика омег в примерном месте предполагаемой синхронизации
    fig_omega_small = plt.figure()
    for agent in range(k_elements):
        if t > 6 and t < len(omega_new[0]) - 5:
            plt.plot(ts[(t-6) * step:(t+4)*step:step], omega_new[agent][t-6 : t+4])
    plt.grid()
    plt.xlabel('t', fontsize=20)
    plt.ylabel('\u03A9', fontsize=20)
    plt.title('Момент синхронизации на графике омеги')

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

    return synchronization_time, sync_last_time, fig, fig_omega_small

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
        path_save, path_save_graphs = mem.save_data([xs, ys, zs, ts], IC, w_arr, [fig_last], ['fig_last_state'], k_elements=k_elements, a=a, tau=tau)
        plt.close()
        mem.draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, 100)

        # Order params
        # find_frequency(xs, ys, zs, ts, w_arr, a)

    # Функция, которая считает время синхронизации
    synchronization_time, omega_fig, fig_omega_small = find_synchronization_time(xs, ys, zs, ts, w_arr, a)

    omega_fig.savefig(path_save + '/fig_last_state' + str(index) +'.png')
    plt.close(omega_fig)
    fig_last.savefig(path_save + '/fig_omega' + str(index) +'.png')
    plt.close(fig_last)
    fig_omega_small.savefig(path_save + '/fig_omega_small' + str(index) +'.png')
    plt.close(fig_omega_small)

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
    synchronization_time, sync_last_time, omega_fig, fig_omega_small = find_synchronization_time(xs, ys, zs, ts, w_arr, a)

    # Сохранить все данные интегрирования, если соло эксперимент
    if isSolo:
        path_save, path_save_graphs = mem.save_data([xs, ys, zs, ts], IC, w_arr, [fig_last], ['fig_last_state'], k_elements=k_elements, a=a, tau=tau)
        plt.close()
        mem.draw_and_save_graphics_many_agents(xs, ys, ts, path_save_graphs, plot_colors, k_elements, 100)
        if not sync_last_time:
            print('Non sync last time')

        plt.plot()

    # Сохраняем графики
    omega_fig.savefig(path_save + '/fig_omega' + str(index) +'.png')
    plt.close(omega_fig)
    fig_last.savefig(path_save + '/fig_last_state' + str(index) +'.png')
    plt.close(fig_last)
    fig_omega_small.savefig(path_save + '/fig_omega_small' + str(index) +'.png')
    plt.close(fig_omega_small)

    return synchronization_time, sync_last_time, [[xs, ys, zs, ts]]

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

    # Конец
    print('Final time: ', time.time() - start_time, end='')
    if tau != 1:
        print('', 'tau:', tau)
    else:
        print('')

def one_elem_omega_a_existance(a, w, IC, k_elements, t_max = s.t_max, tau = s.tau):
    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now())

    func_rossler_2_dim_params = func_rossler_2_dim_params_maker(k_elements)

    w_arr = [w]
    sol = solve_ivp(func_rossler_2_dim_params, [0, t_max], IC, args=(w_arr, a, tau), 
                    rtol=s.toch[0], atol=s.toch[1], method=s.method)
    
    time_after_integrate = time.time()
    print('Integrate time:', time.time() - start_solve_time, 'time:', mem.hms_now())

    xs = sol.y[0]
    ys = sol.y[1]
    zs = sol.y[2]
    ts = sol.t

    return a, [xs, ys, zs, ts]

# 1 agent, grapg omega(a)
def omega_a_experiment_p(IC_fname = 'series_IC_1000_10(1).txt', IC_index = (0, 0), a_inform = (0.15, 0.3, 0.01), w = 1.02, tau = 1):
    start_a, stop_a, step_a = a_inform
    a_arr = np.arange(start_a, stop_a + step_a, step_a)
    num_exps = len(a_arr)

    dir, figs_dir, times_dir = mem.make_dir_for_series_experiments(w, a_arr, num_exps, IC_fname, {"tau": tau})
    IC_arr, w_arr = mem.read_series_IC(s.temporary_path + IC_fname)
    IC_solo_many_agent = IC_arr[IC_index[0]]
    IC = IC_solo_many_agent[IC_index[1] : IC_index[1] + 3]

    s.k_elements = k_elements = 1
    omega_mean_a = []

    n_streams = s.n_streams     # Число потоков для параллельной реализации
    num_batches = - 1 * num_exps // n_streams * - 1 # число пачек экспериментов (число вызова existance)
    batch_step_a = step_a * n_streams

    for exp in range(num_batches):
        a_arr_exp = None
        start = start_a + exp * batch_step_a
        if exp == num_batches - 1:
            if start == stop_a:
                a_arr_exp = [stop_a]
            else:
                a_arr_exp = np.arange(start, stop_a + step_a, step_a)[0:-1]
        else:
            a_arr_exp = np.arange(start, start + batch_step_a - step_a, step_a)

        # Округлить заведомо глупые значения а
        for i in range(len(a_arr_exp)):
            a_arr_exp[i] = round(a_arr_exp[i], 4)

        print(f'-------------------------- Exp a: {a_arr_exp[0]} - {a_arr_exp[-1]} --------------------------')

        # Запускаем параллельно серию одиночных экспериментов
        existance = joblib.Parallel(n_jobs=n_streams)(joblib.delayed(one_elem_omega_a_existance)
                        (a, w, IC, k_elements) for a in a_arr_exp)
        
        for ex_num, ex in enumerate(existance):
            a = ex[0]
            a_str = str(a) + '0' * (5 - len(str(a))) # приводим вывод различных а к одной длине для правильного  порядка
            sol_res = ex[1]
            xs, ys, zs, ts = sol_res

            # plot trajectory
            plt.plot(xs, ys)
            plt.grid()
            plt.title(f'Финальная траектория при a = {a}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(figs_dir + f'/a{a_str}_fig.png')
            plt.close()

            phi, omega, omega_mean = calc_phi_and_omega(xs, ys, zs, ts, w, a)

            # phi(t)
            plt.plot(ts, phi)
            plt.grid()
            plt.xlabel('t')
            plt.ylabel('phi')
            plt.title(f'Зависимость phi(t) при a = {a}')
            plt.savefig(figs_dir + f'/a{a_str}_phi_t.png')
            plt.close()

            # omega(t)
            plt.plot(ts, omega)
            plt.scatter(ts[100], omega[100], color='g')
            plt.grid()
            plt.xlabel('t')
            plt.ylabel('omega')
            plt.title(f'Зависимость omega(t) при a = {a}')
            plt.savefig(figs_dir + f'/a{a_str}_omega_t.png')
            plt.close()

            # Выкидываем первые 100 точек - считаем их переходным процессом
            omega_mean_a.append(np.mean(omega[100:]))
    
    plt.plot(a_arr, omega_mean_a)
    plt.grid()
    plt.xlabel('a')
    plt.ylabel('omega')
    plt.title('Финальная зависимость omega(a)')
    plt.savefig(figs_dir + '/omega_a.png')
    plt.close()


    # for exp_a in np.arange(0, num_exps, n_streams):
    #     print(f'Experiments {exp_a}-{exp_a+n_streams}. ')

    #     # Подбираем номера итераций для всех потоков для нового цикла


    #     # a_existance_step = []
    #     # print(exp_a, step_a, step_a * (n_streams - 1), num_exps * step_a, exp_a + step_a * (n_streams + 1) > num_exps * step_a)
    #     # if exp_a + step_a * (n_streams + 1) > num_exps * step_a:
    #     #     a_existance_step = np.arange(exp_a, num_exps * step_a, step_a)
    #     # else:
    #     #     a_existance_step = np.arange(exp_a, exp_a * n_streamsstop_a)

    #     print(a_existance_step)



################################################################## Make experiments ###########################################################

#path = mem.generate_and_write_series_IC((5., 5., 1.), n_exps=1000, k_elements=k_elements)

#Solo experiment
def series_solo(a_arr = [0.16, 0.22, 0.28], range_ = range(1, 100), IC_path = 'series_IC_1000_10(1).txt'):
    IC_arr, w_arr = mem.read_series_IC(s.temporary_path + IC_path)
    for i in range_:
        print('Exp', i, '---------------------------------------------')
        for a in a_arr:
            s.a = a
            synchronization_time, _, fig = solo_experiment_depend_a_tau_p_2dim(s.a, w_arr, IC_arr, index=i, isSolo=True)
            print(' ', 'Sync time:', synchronization_time, f'a = {s.a}')

# Parallel series
def parallel_series(tau_arr, a_arr = [0.16, 0.22, 0.28], IC_file_name = 'series_IC_1000_10(1).txt'):
    for a in a_arr:
        for tau in tau_arr:
            exp_series_dep_a_tau_p_2dim(s.a, 1000, IC_file_name, tau=tau)

# print(np.arange(0.3, 0.30001, 0.0001)[0:-1])
# omega_a_experiment_p(a_inform=(0.15, 0.3, 0.001))

series_solo(a_arr=[0.22])