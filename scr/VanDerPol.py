from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import memory_worker as mem
import settings as s
import time
import joblib
import itertools
from my_solve_ivp import my_solve
import default_model as dm
import pandas as pd

default_solve_step = 0.001
# method = "Radau"
method = "LSODA"
toch = [1e-12, 1e-12]
dissipative_coup_y_rossler = 0.5

gamma_arr = [0, 2, 4, 6, 8]
delta_arr = [0 for i in range(5)]
mu = 0.002

def phase_converter(phase, type = 1):
    if type == 1:
        if phase > 0:
            if phase > np.pi:
                phase = phase - 2 * np.pi
                phase_converter(phase)
            return phase
        else:
            if phase < - np.pi:
                phase = phase + 2 * np.pi
                phase_converter(phase)
            return phase
    if type == 2:
        if phase < 0:
            phase += 2 * np.pi
        if phase > 2 * np.pi:
            phase -= 2 * np.pi
        return phase_converter(phase)
    
class vdp_params():
    def __init__(self, mu = 0.02, beta = 0.5, gamma = 0., delta = 0.1, l = 0.02, alpha = None):
        self.mu = mu
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
        self.l = l
        if alpha == None:
            alpha = beta # в таком случае получится последняя параллельная связь
        self.alpha = alpha

def solver(params : vdp_params, IC = [1.9, -1.9, 2.1, 2.1], t_max = 40000, default_path = s.grid_experiments_path, 
           solver='solve_ivp', solve_step=default_solve_step, k_elements=2, add_rossler=False):
    prints = True if default_path == s.grid_experiments_path else False

    def func_vdp_2_maker(p : vdp_params, k_elements=2, rossler_params: dm.Rossler_params = dm.Rossler_params()):

        def dissipative_coup(mu, beta, y_this, y_other):
            return mu * beta * (y_other - y_this)
        
        def divide_coup1(mu, l, y_this, y_other):
            return mu * l / (y_other - y_this)
        
        def divide_coup2(mu, l, y_this, y_other):
            return mu * l / (y_this - y_other)
        

        divide_coup = divide_coup2

        def func_vdp_2(t, r):
            x1, y1, x2, y2 = r
            dx1 = y1
            dy1 = p.mu * (1. - x1**2) * y1 - x1 + divide_coup(p.mu, p.l, y1, y2) + dissipative_coup(p.mu, p.beta, y1, y2)
            dx2 = y2
            dy2 = p.mu * (1 + p.gamma - x2**2) * y2 - (1 + p.mu * p.delta) * x2 + divide_coup(p.mu, p.l, y2, y1) + dissipative_coup(p.mu, p.beta, y2, y1)

            return [dx1, dy1, dx2, dy2]
        
        def func_vdp_k_elements(t, r):
            k = 2

            def dissipative_coup_y(r, i, beta = p.beta, mu = p.mu):
                summ = 0
                for j in range(k_elements):
                    if j != i:
                        y_i = r[k*i + 1]
                        y_j = r[k*j + 1]
                        summ += mu * beta * ( y_j - y_i )

                return summ

            res_arr = []
            for agent in range(k_elements):
                x_i = r[k*agent]
                y_i = r[k*agent + 1]
                dx_i = y_i
                coup = dissipative_coup_y(r, agent, p.beta, p.mu)
                dy_i = p.mu * (1. + p.gamma[agent] - x_i**2) * y_i - (1 + p.mu * p.delta[agent]) * x_i + coup
                
                res_arr.append(dx_i)
                res_arr.append(dy_i)
            return res_arr
        
        def func_vdp_k_elements_rossler(t, r):
            k = 2

            def dissipative_coup_y(r, i, beta = p.beta, mu = p.mu):
                summ = 0
                for j in range(k_elements):
                    if j != i:
                        y_i = r[k*i + 1]
                        y_j = r[k*j + 1]
                        summ += mu * beta * ( y_j - y_i )

                return summ
            
            def rossler_coup_y(y_vdp, y_rossler, T):
                return T * (y_rossler - y_vdp)
            
            def rossler_coup_x(x_vdp, x_rossler, T):
                return T * (x_rossler - x_vdp)

            res_arr = []
            x_rossler = r[-3]
            y_rossler = r[-2]
            z_rossler = r[-1]
            for agent in range(k_elements):
                x_i = r[k*agent]
                y_i = r[k*agent + 1]

                dx_i = y_i + rossler_coup_x(x_i, x_rossler, dissipative_coup_y_rossler)
                coup = dissipative_coup_y(r, agent, p.beta, p.mu) + rossler_coup_y(y_i, y_rossler, dissipative_coup_y_rossler)
                dy_i = p.mu * (1. + p.gamma[agent] - x_i**2) * y_i - (1 + p.mu * p.delta[agent]) * x_i + coup
                
                res_arr.append(dx_i)
                res_arr.append(dy_i)

            dx_rossler = - y_rossler - z_rossler
            dy_rossler = x_rossler + 0.16 * y_rossler
            dz_rossler = rossler_params.b + z_rossler * (x_rossler-8.5)
            res_arr.append(dx_rossler)
            res_arr.append(dy_rossler)
            res_arr.append(dz_rossler)

            return res_arr

        # return func_vdp_2
        return func_vdp_k_elements if not add_rossler else func_vdp_k_elements_rossler
    

    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now(), 'l', params.l, 'beta', params.beta)

    rhs = func_vdp_2_maker(params, k_elements)
    # using solve_ivp
    if solver == 'solve_ivp':
        print('Solve function: solve_ivp')
        if solve_step:
            sol = solve_ivp(rhs, [0, t_max], IC, method=method, rtol=toch[0], atol=toch[1], max_step=solve_step)
        else:
            sol = solve_ivp(rhs, [0, t_max], IC, method=method, rtol=toch[0], atol=toch[1])
        xs = [sol.y[0], sol.y[2]]
        ys = [sol.y[1], sol.y[3]]
        ts = sol.t
        print(sol.status)

    # using my solve_ivp
    else:
        print('Solve function: my')
        sol = my_solve(rhs, [0, t_max], IC, h=solve_step)
        ts = sol[0]
        res = sol[1]
        ksi = sol[2]
        xs = [res[0], res[2]]
        ys = [res[1], res[3]]

    if k_elements != 2:
        xs = []
        ys = []
        ts = sol.t
        for i in range(k_elements):
            xs.append(sol.y[i*2])
            ys.append(sol.y[i*2+1])

        if add_rossler:
            xs.append(sol.y[-3])
            ys.append(sol.y[-2])
            zs = sol.y[-1]
            return xs, ys, zs, ts
        return xs, ys, ts

    time_after_integrate = time.time()
    if prints:
        print(f'Integrate time: {(time.time() - start_solve_time):0.1f}s', 'time:', mem.hms_now())

    size = len(ts)

    def create_and_mk_dir(dir):
        counter = 1
        new_dir = dir

        import os
        while os.path.exists(new_dir):
            new_dir = f'{dir}({counter})'
            counter += 1

        mem.make_dir(new_dir)
        return new_dir

    dir_name = f"vdp_b_{params.beta}_l_{params.l}_d_{params.delta}_{t_max}_"
    dir_name += f"{'s' if solver == 'solve_ivp' else 'm'}{f'_{solve_step}' if solve_step !=  default_solve_step else ''}"
    dir_name += f"{f'_{method}' if method != 'RK45' else ''}"
    print('Dir name:', dir_name)
    dir_path = default_path + dir_name
    
    dir_path = create_and_mk_dir(dir_path)

    def plot_timeline_graph(x : list, t : list, ylabel : str, save_name : str = '', path_save : str = dir_path, 
                              figsize_ : list=[12, 3], xlabel : list='t', xlims : list = [0, t_max],
                              subplots_adjust : list=[0.11, 0.97, 0.225, 0.92], 
                              format : str='.png', x2 : list=[], font_size : int = 18) -> None:
            plt.figure(figsize=figsize_)
            plt.subplots_adjust(left=subplots_adjust[0], right=subplots_adjust[1], bottom=subplots_adjust[2], top=subplots_adjust[3])
            if len(x2) == 0:
                plt.plot(t, x)
            else:
                plt.plot(t, x, label='1')
                plt.plot(t, x2, label='2')
                plt.legend()
            plt.grid()
            plt.xlabel(xlabel, fontsize=font_size)
            plt.ylabel(ylabel, fontsize=font_size)
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.xlim(xlims[0], xlims[1])
            if save_name == '':
                plt.show()
            else:
                plt.savefig(path_save + '/' + save_name + '.png')
            plt.close()

    # t_100 = np.searchsorted(ts, 100)
    # plot_timeline_graph(xs[0][:t_100], ts[:t_100], 'x', 'vdp_xt_first_100', x2=xs[1][:t_100], xlims=[0, 100])
    # plot_timeline_graph(ys[0][:t_100], ts[:t_100], 'y', 'vdp_yt_first_100', x2=ys[1][:t_100], xlims=[0, 100])


    # t_min_100 = np.searchsorted(ts, t_max-100)
    # plot_timeline_graph(xs[0][t_min_100:], ts[t_min_100:], 'x', 'vdp_xt_last_100', x2=xs[1][t_min_100:], xlims=[t_max-100, t_max])
    # plot_timeline_graph(ys[0][t_min_100:], ts[t_min_100:], 'y', 'vdp_yt_last_100', x2=ys[1][t_min_100:], xlims=[t_max-100, t_max])

    t_last = 200
    t_last_200 = np.searchsorted(ts, t_max-t_last)
    plot_timeline_graph(xs[0][t_last_200:], ts[t_last_200:], 'x', f'vdp_xt_last_{t_last}', x2=xs[1][t_last_200:], xlims=[t_max-t_last, t_max])
    plot_timeline_graph(ys[0][t_last_200:], ts[t_last_200:], 'y', f'vdp_yt_last_{t_last}', x2=ys[1][t_last_200:], xlims=[t_max-t_last, t_max])
    t_200 = np.searchsorted(ts, t_last)
    plot_timeline_graph(xs[0][:t_200], ts[:t_200], 'x', f'vdp_xt_first_{t_last}', x2=xs[1][:t_200], xlims=[0, t_last])
    plot_timeline_graph(ys[0][:t_200], ts[:t_200], 'y', f'vdp_yt_first_{t_last}', x2=ys[1][:t_200], xlims=[0, t_last])

    plt.figure(figsize=[5,5])
    plt.subplots_adjust(0.2, 0.2, 0.925, 0.925)
    
    ind_last = np.searchsorted(ts, t_max - t_last)
    plt.plot(xs[0][ind_last:], ys[0][ind_last:], label='1')
    plt.plot(xs[1][ind_last:], ys[1][ind_last:], label='2')
    plt.scatter(xs[0][-1], ys[0][-1])
    plt.scatter(xs[1][-1], ys[1][-1])
    plt.legend()
    plt.grid()
    # plt.xlim(-2.3, 2.3)
    # plt.ylim(-2.3, 2.3)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(dir_path + '/' + f'xy_last_{t_last}.png')
    # plt.show()
    plt.close()

    plt.figure(figsize=[5,5])
    plt.subplots_adjust(0.2, 0.2, 0.925, 0.925)
    plt.plot(xs[0], ys[0], label='1')
    plt.plot(xs[1], ys[1], label='2')
    plt.scatter(xs[0][-1], ys[0][-1])
    plt.scatter(xs[1][-1], ys[1][-1])
    plt.legend()
    plt.grid()
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(dir_path + '/' + f'xy_last_full.png')
    plt.close()

    try:
        plot_timeline_graph(ksi, ts[1:], r'$\xi(t)$', 'xi_t')
    except Exception as e:
        print(e)

    # Dists
    dists = []
    for i in range(size):
        dists.append(np.sqrt((xs[0][i]-xs[1][i])**2+(ys[0][i]-ys[1][i])**2))
    plot_timeline_graph(dists, ts, 'd', 'vdp_dist')

    # phases diff
    phases_diff = []
    for i in range(size):
        phase1 = np.arctan2(ys[0][i], xs[0][i])
        phase2 = np.arctan2(ys[1][i], xs[1][i])
        phases_diff.append(phase_converter(phase2 - phase1, type=1))
    plot_timeline_graph(phases_diff, ts, r'$\phi_2 - \phi_1$', 'vdp_phases_diff')
    plot_timeline_graph(phases_diff[t_last_200:], ts[t_last_200:], r'$\phi_2 - \phi_1$', f'vdp_phases_diff_last_{t_last}', xlims=[t_max-t_last, t_max])

    # amplitudes diff
    As_diff = []
    As = [[], []]
    for i in range(size):
        A1 = np.sqrt(xs[0][i]**2 + ys[0][i]**2)
        A2 = np.sqrt(xs[1][i]**2 + ys[1][i]**2)
        As[0].append(A1)
        As[1].append(A2)
        As_diff.append(A1 - A2)
    plot_timeline_graph(As_diff, ts, r'$A_1 - A_2$', 'vdp_As_diff')
    plot_timeline_graph(As[0], ts, r'$A_1, A_2$', 'vdp_As', x2=As[1])
    plot_timeline_graph(As_diff[t_last_200:], ts[t_last_200:], r'$A_1 - A_2$', f'vdp_As_diff_last_{t_last}', xlims=[t_max-t_last, t_max])
    plot_timeline_graph(As[0][t_last_200:], ts[t_last_200:], r'$A_1, A_2$', f'vdp_As_last_{t_last}', x2=As[1][t_last_200:], xlims=[t_max-t_last, t_max])

    # Считаем метрики
    transition_process = t_max * 0.9
    transition_process_time = np.searchsorted(ts, transition_process)
    diff_phases_average = np.mean(phases_diff[transition_process_time:])
    diff_As_average = np.mean(As_diff[transition_process_time:])
    diff_phases_max = np.max(np.abs(phases_diff[transition_process_time:]))
    diff_As_max = np.max(np.abs(As_diff[transition_process_time:]))
    # записываем
    with open(dir_path + '/metrix.txt', 'w') as f:
        print('transition process:', transition_process, file=f)
        print('diff phases avg:', diff_phases_average, file=f)
        print('diff As avg:', diff_As_average, file=f)
        print('Params', file=f)
        print('mu:', params.mu, file=f)
        print('beta', params.beta, file=f)
        print('delta', params.delta, file=f)
        print('gamma', params.gamma, file=f)
        print('l', params.l, file=f)
        print('alpha', params.alpha, file=f)
        print('IC', IC, file=f)
        print('method', method, file=f)
        print('toch', toch, file=f)

    

    return params.l, params.beta, diff_phases_average, diff_phases_max, diff_As_average, diff_As_max

def solver_joblib(l, beta, t_max, default_path):
    beta = round(beta, 3)
    l = round(l, 3)
    params = vdp_params(beta=beta, l=l)
    res = solver(params, t_max = t_max, default_path=default_path)
    return res

def dep_of_l_beta():
    betas = np.arange(0.1, 1.11, 0.01)
    ls = np.arange(0.01, 0.51, 0.01)
    print(betas, ls)
    n_streams = 64
    t_max = 40000

    time = mem.hms_now().replace(':', '-')
    path = mem.make_dir(s.grid_experiments_path + f'vdp_{time}')
    path += '/'

    existance = joblib.Parallel(n_jobs=n_streams)(joblib.delayed(solver_joblib)(l, beta, t_max, path) for l, beta in itertools.product(ls, betas))

    with open(path + 'res.txt', 'w') as f:
        for ex in existance:
            for e in ex:
                print(e, file=f, end=' ')
            print('', file=f)

def series_experiments_dep_delta_l():
    l_arr = np.arange(0.02, 0.21, 0.01)
    delta_arr = np.arange(0., 0.006, 0.001)
    print(l_arr, delta_arr)

    time = mem.hms_now().replace(':', '-')
    path = mem.make_dir(s.grid_experiments_path + f'vdp_l_delta{time}')
    path += '/'
    for l in l_arr:
        for delta in delta_arr:
            l = round(l, 4)
            delta = round(delta, 4)
            print('l', l, 'delta', delta)
            params = vdp_params(l=l, beta=0., alpha=0., delta=delta, mu=0.02)
            solver(params, t_max=4000, solver='my', solve_step=0.01, IC=[2.1, 0.1, 1.5, 0.], default_path=path)

def solo():
    # params = vdp_params(l=0.1, beta=.0, alpha=.0, delta=0.02, mu=0.02)

    # l_arr = np.arange(0.01, 0.21, 0.01)
    # print(l_arr)
    # for l in l_arr:
    #     l = round(l, 5)
    #     params = vdp_params(l=l, beta=0.5, alpha=0.0, delta=0.1, mu=0.001)
    #     solver(params, t_max=40000, solver='solve_ivp', solve_step=0.05, IC=[0, 1.9, 0, 2.1])

    # для двух элементов
    params = vdp_params(l=0., beta=0.1, alpha=0.0, delta=0.0, mu=0.01, gamma=0.1)
    solver(params, t_max=6000, solver='solve_ivp', solve_step=None, IC=[0.1, 1.8, 0, 2.2])

def solo_many_agents():
    # Для N Элементов
    k_elements = 5
    # gamma_arr = [0, 0.3, 0.6, 0.9, 1.2]
    # gamma_arr = [0 for i in range(k_elements)]
    # delta_arr = [0, 0.1, 0.2, 0.3, 0.4]
    params = vdp_params(l=0., beta=0.4, alpha=0.0, delta=delta_arr, mu=mu, gamma=gamma_arr)
    IC = [
        0.1, 1.8,
        0.2, 1.9, 
        0.0, 2.2,
        0.2, 2.1,
        0.4, 2.1
    ]
    t_max = 10000
    global method
    method = "LSODA"
    xs, ys, ts = solver(params, t_max=t_max, solver='solve_ivp', solve_step=None, IC=IC, k_elements=5)
    ts_last_100 = np.searchsorted(ts, t_max-100)
    plot_colors = mem.make_colors(k_elements)

    xs_df = pd.DataFrame(xs)
    xs_df.to_csv(f"{s.grid_experiments_path}vdp_special/before_add_agent_xs.txt", sep=' ', header=False, index=False)
    ys_df = pd.DataFrame(ys)
    ys_df.to_csv(f"{s.grid_experiments_path}vdp_special/before_add_agent_ys.txt", sep=' ', header=False, index=False)
    ts_df = pd.DataFrame(ts)
    ts_df.to_csv(f"{s.grid_experiments_path}vdp_special/before_add_agent_ts.txt", sep=' ', header=False, index=False)

    plt.figure(figsize=[4,4])
    for agent in range(k_elements-1, -1, -1):
        plt.plot(xs[agent][ts_last_100:], ys[agent][ts_last_100:], color=plot_colors[agent], zorder = 1, label=f"Agent {agent}")
        plt.scatter(xs[agent][-1], ys[agent][-1], color=plot_colors[agent], zorder=2)
    # plt.grid()
    fontsize = 20
    plt.xlabel('x', fontsize=fontsize)
    plt.ylabel('y', fontsize=fontsize)
    plt.legend()
    plt.xlim(-2.4, 2.4)
    plt.ylim(-2.4, 2.4)
    plt.xticks([-3, -2, -1, 0, 1, 2, 3], fontsize=fontsize)
    plt.yticks([-2, -1, 0, 1, 2], fontsize=fontsize)
    plt.axis('equal')
    plt.subplots_adjust(0.2, 0.2, 0.95, 0.95)
    plt.show()

def solo_many_agents_rossler(IC_vdp = None, IC_rossler = None):
        # Для N Элементов
    k_elements = 5
    # gamma_arr = [0, 0.3, 0.6, 0.9, 1.2]
    # gamma_arr = [0 for i in range(k_elements)]
    # delta_arr = [0, 0.1, 0.2, 0.3, 0.4]
    params = vdp_params(l=0., beta=0.2, alpha=0.0, delta=delta_arr, mu=mu, gamma=gamma_arr)
    global dissipative_coup_y_rossler
    dissipative_coup_y_rossler = 8
    if not IC_vdp:
        IC_vdp = [
            0.1, 1.8,
            0.2, 1.9, 
            0.0, 2.2,
            0.2, 2.1,
            0.4, 2.1,
        ]
    if not IC_rossler:
        IC_rossler = [
            1, 1, 0.1,
        ]

    IC = IC_vdp[0:2*k_elements] + IC_rossler
    t_max = 1000
    global method
    method = "LSODA"
    xs, ys, zs, ts = solver(params, t_max=t_max, solver='solve_ivp', solve_step=None, IC=IC, k_elements=k_elements, add_rossler=True)
    ts_last_100 = np.searchsorted(ts, t_max-100)
    plot_colors = mem.make_colors(k_elements)

    xs_df = pd.DataFrame(xs)
    xs_df.to_csv(f"{s.grid_experiments_path}vdp_special/with_rossler_xs.txt", sep=' ', header=False, index=False)
    ys_df = pd.DataFrame(ys)
    ys_df.to_csv(f"{s.grid_experiments_path}vdp_special/with_rossler_ys.txt", sep=' ', header=False, index=False)
    zs_df = pd.DataFrame(zs)
    zs_df.to_csv(f"{s.grid_experiments_path}vdp_special/with_rossler_zs.txt", sep=' ', header=False, index=False)
    ts_df = pd.DataFrame(ts)
    ts_df.to_csv(f"{s.grid_experiments_path}vdp_special/with_rossler_ts.txt", sep=' ', header=False, index=False)

    def create_and_mk_dir(dir):
        counter = 1
        new_dir = dir

        import os
        while os.path.exists(new_dir):
            new_dir = f'{dir}({counter})'
            counter += 1

        mem.make_dir(new_dir)
        return new_dir

    dir_name = f"vdp_add_rossler_t_{t_max}_coup_{dissipative_coup_y_rossler}"
    dir_name = create_and_mk_dir(s.grid_experiments_path + dir_name)
    data_path = mem.make_dir(dir_name + '/data')
    plot_colors.append('red')
    # mem.draw_and_save_graphics_many_agents(xs, ys, ts, data_path, plot_colors=plot_colors, _k_elements = k_elements+1, t_step=2)

    # plt.figure(figsize=[6, 6])
    # plt.plot(xs[-1][ts_last_100:], ys[-1][ts_last_100:], color='red', label='Agent Rossler', zorder=1)
    # plt.scatter(xs[-1][-1], ys[-1][-1], color='red')
    # for agent in range(k_elements):
    #     plt.plot(xs[agent][ts_last_100:], ys[agent][ts_last_100:], color=plot_colors[agent], zorder = 1, label=f"Agent {agent}")
    #     plt.scatter(xs[agent][-1], ys[agent][-1], color=plot_colors[agent], zorder=2)
    # # plt.grid()
    # fontsize = 20
    # plt.xlabel('x', fontsize=fontsize)
    # plt.ylabel('y', fontsize=fontsize)
    # plt.legend()
    # # plt.xlim(-2.4, 2.4)
    # # plt.ylim(-2.4, 2.4)
    # plt.xticks(fontsize=fontsize)
    # plt.yticks(fontsize=fontsize)
    # plt.axis('equal')
    # plt.subplots_adjust(0.2, 0.2, 0.95, 0.95)
    # plt.savefig(dir_name + '/xy_last_100.png')
    # plt.show()

    t_end = 30
    fontsize=20
    t_end_ind = np.searchsorted(ts, t_end)
    plt.figure(figsize=[8,4])
    for agent in range(len(xs)):
        plt.plot(ts[:t_end_ind], xs[agent][:t_end_ind], color=plot_colors[agent])
    plt.xlabel(r'$t$', fontsize=fontsize)
    plt.ylabel(r'$x$', fontsize=fontsize)
    plt.subplots_adjust(0.15, 0.22, 0.98, 0.96)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()

def vdp_rossler_controls():

    global gamma_arr, delta_arr
    delta_arr = [0 for i in range(5)]
    gamma_arr = [0, 3, 6, 9, 12]
    
    solo_many_agents()
    xs_before = pd.read_table(f"{s.grid_experiments_path}vdp_special/before_add_agent_xs.txt", sep=' ', header=None).values
    ys_before = pd.read_table(f"{s.grid_experiments_path}vdp_special/before_add_agent_ys.txt", sep=' ', header=None).values
    ts_before = pd.read_table(f"{s.grid_experiments_path}vdp_special/before_add_agent_ts.txt", sep=' ', header=None).values
    ts_before = [item[0] for item in ts_before]
    k_elements = len(xs_before)

    IC_next = []
    for agent in range(k_elements):
        IC_next.append(xs_before[agent][-1])
        IC_next.append(ys_before[agent][-1])
    
    IC_rossler = [-0.5, -3., 0.]
    solo_many_agents_rossler(IC_vdp=IC_next, IC_rossler=IC_rossler)

    xs_after = pd.read_table(f"{s.grid_experiments_path}vdp_special/with_rossler_xs.txt", sep=' ', header=None).values
    ys_after = pd.read_table(f"{s.grid_experiments_path}vdp_special/with_rossler_ys.txt", sep=' ', header=None).values
    zs_after = pd.read_table(f"{s.grid_experiments_path}vdp_special/with_rossler_zs.txt", sep=' ', header=None).values
    ts_after = pd.read_table(f"{s.grid_experiments_path}vdp_special/with_rossler_ts.txt", sep=' ', header=None).values
    ts_after = [item[0] for item in ts_after]

    plot_colors = mem.make_colors(len(xs_before))
    plot_colors.append('red')
    t_max_before = ts_before[-1]
    t_max_after = ts_after[-1]
    start_half_before = t_max_before - 20
    i_half_before = np.searchsorted(ts_before, start_half_before)
    start_half_after = 180
    i_half_after = np.searchsorted(ts_after, start_half_after)
    ts_half_before = [item - start_half_before for item in ts_before[i_half_before:]]
    ts_half_after = [item + 20 for item in ts_after[:i_half_after]]
    plt.figure(figsize = [8, 4])
    for agent in range(k_elements):
        plt.plot(ts_half_before, xs_before[agent][i_half_before:], color = plot_colors[agent])
        plt.plot(ts_half_after, xs_after[agent][:i_half_after], color=plot_colors[agent], label=f"Agent {agent + 1}")
    plt.plot(ts_half_after, xs_after[-1][:i_half_after], color=plot_colors[-1], label="Agent Rossler")
    # plt.text(20, 1, r'$t_{cr}$', 
    #      ha='center', va='top', fontsize=12)
    # Получаем текущие метки по оси X
    
    plt.xticks([0, 50, 100, 150, 200], fontsize=20)
    plt.yticks([15, 10, 5, 0, -5, -10, -15], fontsize=20)
    ax = plt.gca()
    ticks = list(ax.get_xticks())
    labels = [str(tick) for tick in ticks]

    # Добавляем t=20 в метки, если его там нет
    t_cr = 20
    if t_cr not in ticks:
        ticks.append(t_cr)
        labels.append(r'$t_{cr}$')

    # Устанавливаем новые метки и подписи
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    plt.xlabel(r'$t$', fontsize=20)
    plt.ylabel(r'$x$', fontsize=20)
    plt.legend(loc="upper right")
    plt.subplots_adjust(0.13, 0.2, 0.97, 0.95)
    plt.show()

    i_half_after_rossler = np.searchsorted(ts_after, 280)
    ts_half_after_rossler = [item + 20 for item in ts_after[:i_half_after_rossler]]
    plt.figure(figsize = [8, 4])
    for agent in range(k_elements):
        plt.plot(ts_half_after_rossler, xs_after[agent][:i_half_after_rossler], color=plot_colors[agent], label=f"Agent {agent + 1}")
    plt.plot(ts_half_after_rossler, xs_after[-1][:i_half_after_rossler], color=plot_colors[-1], label=f"Agent Rossler")
    plt.xlabel(r'$t$', fontsize=20)
    plt.ylabel(r'$x$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    plt.subplots_adjust(0.12, 0.2, 0.97, 0.95)
    plt.show()

if __name__ == '__main__':
    # solo()
    # series_experiments_dep_delta_l()
    # solo_many_agents_rossler()
    # solo_many_agents

    vdp_rossler_controls()


# Получить результаты параллельного движения при двух связях (а не при трех как было раньше)