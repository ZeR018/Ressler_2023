from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import memory_worker as mem
import colorama
import settings as s
import time
import joblib
import itertools

colorama.init()

def phase_converter(phase):
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
    
class vdp_params():
    def __init__(self, mu = 0.001, beta = 0.5, gamma = 0., delta = 0.1, l = 0.02):
        self.mu = mu
        self.delta = delta
        self.beta = beta
        self.gamma = gamma
        self.l = l

def solver(params : vdp_params, IC = [1.9, -1.9, 2.1, 2.1], t_max = 40000, default_path = s.grid_experiments_path):
    prints = True if default_path == s.grid_experiments_path else False

    def func_vdp_2_maker(p : vdp_params):

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
            # dy1 = p.mu * (1. - x1**2) * y1 - x1 - dissipative_coup(p.mu, p.beta, y2, y1)
            # dy2 = p.mu * (1 + p.gamma - x2**2) * y2 - (1 + p.mu * p.delta) * x2 - dissipative_coup(p.mu, p.beta, y1, y2)
            return [dx1, dy1, dx2, dy2]
        return func_vdp_2
    
    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now(), 'l', params.l, 'beta', params.beta)

    # print('Start')
    rhs = func_vdp_2_maker(params)
    sol = solve_ivp(rhs, [0, t_max], IC, method='RK45', rtol=1e-4, atol=1e-4, max_step=0.05)
    if prints:
        print('Integrate status:', sol.status)

    time_after_integrate = time.time()
    if prints:
        print(f'Integrate time: {(time.time() - start_solve_time):0.1f}s', 'time:', mem.hms_now())

    xs = [sol.y[0], sol.y[2]]
    ys = [sol.y[1], sol.y[3]]
    ts = sol.t
    
    size = len(ts)

    dir_path = default_path + f'vdp_b_{params.beta}_l_{params.l}_{t_max}'
    def create_and_mk_dir(dir):
        counter = 1
        new_dir = dir

        import os
        while os.path.exists(new_dir):
            new_dir = f'{dir}({counter})'
            counter += 1

        mem.make_dir(new_dir)
        return new_dir
    
    create_and_mk_dir(dir_path)

    def plot_timeline_graph(x : list, t : list, ylabel : str, save_name : str = '', path_save : str = dir_path, 
                              figsize_ : list=[12, 3], xlabel : list='t', xlims : list = [0, t_max],
                              subplots_adjust : list=[0.11, 0.97, 0.225, 0.97], 
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

    t_100 = np.searchsorted(ts, 100)
    plot_timeline_graph(xs[0][:t_100], ts[:t_100], 'x', 'vdp_xt_first_100', x2=xs[1][:t_100], xlims=[0, 100])
    plot_timeline_graph(ys[0][:t_100], ts[:t_100], 'y', 'vdp_yt_first_100', x2=ys[1][:t_100], xlims=[0, 100])


    t_min_100 = np.searchsorted(ts, t_max-100)
    plot_timeline_graph(xs[0][t_min_100:], ts[t_min_100:], 'x', 'vdp_xt_last_100', x2=xs[1][t_min_100:], xlims=[t_max-100, t_max])
    plot_timeline_graph(ys[0][t_min_100:], ts[t_min_100:], 'y', 'vdp_yt_last_100', x2=ys[1][t_min_100:], xlims=[t_max-100, t_max])


    # graph xy last 100
    plt.figure(figsize=[5,5])
    plt.plot(xs[0][t_min_100:], ys[0][t_min_100:], label='1')
    plt.plot(xs[1][t_min_100:], ys[1][t_min_100:], label='2')
    plt.scatter(xs[0][-1], ys[0][-1])
    plt.scatter(xs[1][-1], ys[1][-1])
    plt.legend()
    plt.grid()
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(dir_path + '/' + f'xy_last_{t_min_100}.png')

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
        phases_diff.append(phase_converter(phase1 - phase2))
    plot_timeline_graph(phases_diff, ts, r'$\phi_1 - \phi_2$', 'vdp_phases_diff')

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

    # Считаем метрики
    # transition_process = int(t_max / 2)
    # transition_process = 10000
    # if t_max < 10000:
    #     transition_process = int(t_max / 2)
    # elif t_max >= 25000:
    #     transition_process = 15000
    # elif t_max > 30000:
    #     transition_process = 20000
    after_transition_process = 10000
    transition_process = t_max - after_transition_process

    transition_process_time = np.searchsorted(ts, transition_process)
    diff_phases_average = np.mean(phases_diff[transition_process_time:])
    diff_As_average = np.mean(As_diff[transition_process_time:])
    diff_phases_max = np.max(np.abs(phases_diff[transition_process_time:]))
    diff_As_max = np.max(np.abs(As_diff[transition_process_time:]))
    # записываем
    with open(dir_path + '/metrix.txt', 'w') as f:
        print('transition process:', transition_process, file=f)
        print('diff phases avg:', diff_phases_average, file=f)
        print('diff phases max:', diff_phases_max, file=f)
        print('diff As avg:', diff_As_average, file=f)
        print('diff As max:', diff_As_max, file=f)

    with open(dir_path + '/As_phases.txt', 'w') as f:
        for i in range(after_transition_process):
            t_i = transition_process + i
            ind_start = np.searchsorted(ts, t_i)
            ind_end = np.searchsorted(ts, t_i + 1)
            print(np.mean(phases_diff[ind_start:ind_end]), np.mean(As_diff[ind_start:ind_end]), file=f)

    return params.l, params.beta, diff_phases_average, diff_phases_max, diff_As_average, diff_As_max

def solver_joblib(l, beta, t_max, default_path):
    beta = round(beta, 3)
    l = round(l, 3)
    params = vdp_params(beta=beta, l=l)
    res = solver(params, t_max = t_max, default_path=default_path)
    return res

def parallel():
    betas = np.arange(0.3, 1.11, 0.01)
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

if __name__ == '__main__':
    params = vdp_params(beta=0.78, l=0.02)
    solver(params, t_max = 20000)
