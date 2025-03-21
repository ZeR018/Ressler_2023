from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import memory_worker as mem
import colorama
import settings as s
import time
import joblib
import itertools
from my_solve_ivp import my_solve

colorama.init()
default_solve_step = 0.001

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

def solver(params : vdp_params, IC = [1.9, -1.9, 2.1, 2.1], t_max = 40000, default_path = s.grid_experiments_path, solver='solve_ivp', solve_step=default_solve_step):
    prints = True if default_path == s.grid_experiments_path else False

    print('l',params.l, 'b', params.beta, 'a', params.alpha, 'd', params.delta)

    def func_vdp_2_maker(p : vdp_params):

        def dissipative_coup(mu, beta, y_this, y_other):
            return mu * beta * (y_other - y_this)
        
        def divide_coup1(mu, l, y_this, y_other):
            return mu * l / (y_other - y_this)
        
        def divide_coup2(mu, l, y_this, y_other):
            return mu * l / (y_this - y_other)
        
        def conservative_coup(mu, alpha, x_this, x_other):
            return mu * alpha * (x_other - x_this)
        

        divide_coup = divide_coup2

        def func_vdp_2(t, r):
            # if prints:
            #     print(f'\033[F\033[KCurrent integrate time: {round(t, 1)};')
            x1, y1, x2, y2 = r

            coups_1 = conservative_coup(p.mu, p.alpha, x1, x2) + divide_coup(p.mu, p.l, y1, y2) + dissipative_coup(p.mu, p.beta, y1, y2) + \
                conservative_coup(p.mu, p.alpha, x1, x2)
            coups_2 = conservative_coup(p.mu, p.alpha, x2, x1) + divide_coup(p.mu, p.l, y2, y1) + dissipative_coup(p.mu, p.beta, y2, y1) + \
                conservative_coup(p.mu, p.alpha, x2, x1)
            
            dx1 = y1
            dy1 = p.mu * (1. - x1**2) * y1 - x1 + coups_1
            dx2 = y2
            dy2 = p.mu * (1 + p.gamma - x2**2) * y2 - (1 + p.mu * p.delta) * x2 + coups_2
            
            return [dx1, dy1, dx2, dy2]
        return func_vdp_2
    
    # Integrate
    start_solve_time = time.time()
    print('Start solve time:', mem.hms_now(), 'l', params.l, 'beta', params.beta)

    rhs = func_vdp_2_maker(params)
    solve_step = 0.0005
    # using solve_ivp
    if solver == 'solve_ivp':
        print('Solve function: solve_ivp')
        sol = solve_ivp(rhs, [0, t_max], IC, method='RK45', rtol=1e-4, atol=1e-4, max_step=solve_step)
        xs = [sol.y[0], sol.y[2]]
        ys = [sol.y[1], sol.y[3]]
        ts = sol.t

    # using my solve_ivp
    else:
        print('Solve function: my')
        sol = my_solve(rhs, [0, t_max], IC, h=solve_step)
        ts = sol[0]
        res = sol[1]
        ksi = sol[2]
        xs = [res[0], res[2]]
        ys = [res[1], res[3]]

    time_after_integrate = time.time()
    if prints:
        print(f'Integrate time: {(time.time() - start_solve_time):0.1f}s', 'time:', mem.hms_now())


    
    size = len(ts)

    dir_path = default_path + f"vdp_b_{params.beta}_l_{params.l}_d_{params.delta}_{t_max}_"
    dir_path += f"{'s' if solver == 'solve_ivp' else 'm'}{f'_solve_step' if solve_step !=  default_solve_step else ''}"
    try:
        mem.make_dir(dir_path)
    except Exception:
        print('Dir is already exists') 
        tmp_dir_name = dir_path + '-'
        mem.rename_dir(dir_path, tmp_dir_name)
        mem.rename_dir(tmp_dir_name, dir_path)

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

    t_100 = np.searchsorted(ts, 100)
    plot_timeline_graph(xs[0][:t_100], ts[:t_100], 'x', 'vdp_xt_first_100', x2=xs[1][:t_100], xlims=[0, 100])
    plot_timeline_graph(ys[0][:t_100], ts[:t_100], 'y', 'vdp_yt_first_100', x2=ys[1][:t_100], xlims=[0, 100])


    t_min_100 = np.searchsorted(ts, t_max-100)
    plot_timeline_graph(xs[0][t_min_100:], ts[t_min_100:], 'x', 'vdp_xt_last_100', x2=xs[1][t_min_100:], xlims=[t_max-100, t_max])
    plot_timeline_graph(ys[0][t_min_100:], ts[t_min_100:], 'y', 'vdp_yt_last_100', x2=ys[1][t_min_100:], xlims=[t_max-100, t_max])

    plt.figure(figsize=[5,5])
    plt.subplots_adjust(0.2, 0.2, 0.925, 0.925)
    t_last = 100
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
        phases_diff.append(phase_converter(phase2 - phase1, type=2))
    plot_timeline_graph(phases_diff, ts, r'$\phi_2 - \phi_1$', 'vdp_phases_diff')

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


def main():
    # params = vdp_params(l=0.1, beta=.0, alpha=.0, delta=0.02, mu=0.02)
    params = vdp_params(l=0.1, beta=0., alpha=.0, delta=0.02, mu=0.02)
    solver(params, t_max=20000, solver='my')


if __name__ == '__main__':
    main()

# Посчитать все случаи из отчета с новыми mu, delta
# Аналитический прикол сделать
# Посчитать мгновенные и средние частоты
# Написать свой численный метод
# Посмотреть зависимость от дельты при связи в знаменателе
