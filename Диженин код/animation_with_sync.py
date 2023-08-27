import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import hilbert

import math

def num_lorenc(X1_n, Y1_n, Z1_n, h, sigma, ro, Beta, numsteps):
    x1_n1 = X1_n
    y1_n1 = Y1_n
    z1_n1 = Z1_n

    a = 1.4
    for k in range(numsteps-1):
        dx_1 = sigma * (y1_n1[k] - x1_n1[k])
        dy_1 = x1_n1[k] * (ro - z1_n1[k]) - y1_n1[k]
        dz_1 = x1_n1[k] * y1_n1[k] - Beta * z1_n1[k]
        # dx_1 = (-a * x1_n1[k] - 4 * y1_n1[k] - 4 * z1_n1[k] - pow(y1_n1[k], 2))
        # dy_1 = (-a * y1_n1[k] - 4 * z1_n1[k] - 4 * x1_n1[k] - pow(z1_n1[k], 2))
        # dz_1 = (-a * z1_n1[k] - 4 * x1_n1[k] - 4 * y1_n1[k] - pow(x1_n1[k], 2))
        # dx_1 = 0
        # dy_1 = 0
        # dz_1 = 0

        x1_n1[k+1] = x1_n1[k] + h * dx_1
        y1_n1[k+1] = y1_n1[k] + h * dy_1
        z1_n1[k+1] = z1_n1[k] + h * dz_1

    return x1_n1, y1_n1, z1_n1

def num_rossler_together(x1_n, y1_n, z1_n, h, a, b, c, w1, d, numsteps, X, Y, Z):
    def d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n):
        d_val = 0.0
        if (pow(x1_n - x2_n, 2) + pow(y1_n - y2_n, 2) + pow(z1_n - z2_n, 2) < pow(1, 2)):
            d_val = 0.02
        # if (pow(x1_n - x2_n, 2) + pow(y1_n - y2_n, 2) + pow(z1_n - z2_n, 2) < pow(2, 2)):
        #     d_val = 0.02

        return d_val
    
    def D(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n):
        d_val = 0.0
        if (pow(x1_n - x2_n, 2) + pow(y1_n - y2_n, 2) + pow(z1_n - z2_n, 2) < pow(0.9, 2)):
            d_val = 1
        # if (pow(x1_n - x2_n, 2) + pow(y1_n - y2_n, 2) + pow(z1_n - z2_n, 2) < pow(2, 2)):
        #     d_val = 0.02

        return d_val

    # --- connection : d_it = {0.2 or 0}

    x1_n1 = x1_n
    y1_n1 = y1_n
    z1_n1 = z1_n

    for i in range(5):
        for k in range(numsteps-1):
            d_fin_x_1 = 0
            d_fin_x_2 = 0
            d_fin_z_1 = 0
            d_fin_z_2 = 0
            d_fin_y = 0
            d_fin_z = 0

            d_fin_D = 0
            DX_sum = 0
            DY_sum = 0
            DZ_sum = 0
            for j in range(5):
                d_res = d(x1_n1[i][k], x1_n1[j][k], y1_n1[i][k], y1_n1[j][k], z1_n1[i][k], z1_n1[j][k])
                if (j != i):
                    # d_res = d(x1_n1[i][k], x1_n1[j][k], y1_n1[i][k], y1_n1[j][k], z1_n1[i][k], z1_n1[j][k])
                    d_fin_x_1  = d_fin_x_1 + d_res * (x1_n1[j][k] - x1_n1[i][k])
                    d_fin_x_2  = d_fin_x_2 + d_res / (x1_n1[i][k] - x1_n1[j][k])
                    d_fin_z_1  = d_fin_z_1 + d_res * (z1_n1[j][k] - z1_n1[i][k])
                    d_fin_z_2  = d_fin_z_2 + d_res / (z1_n1[i][k] - z1_n1[j][k])
                    d_fin_y    = d_fin_y + d_res * (y1_n1[j][k] - y1_n1[i][k])
                    d_fin_z    = d_fin_z + d_res * (z1_n1[j][k] - z1_n1[i][k])

                    d_fin_D = d_fin_D + d(x1_n1[i][k], X[k], y1_n1[i][k], Y[k], z1_n1[i][k], Z[k]) * (Y[k] - y1_n1[i][k])
                    DX_sum = DX_sum + d(x1_n1[i][k], X[k], y1_n1[i][k], Y[k], z1_n1[i][k], Z[k]) * (X[k] - x1_n1[i][k])
                    DY_sum = DY_sum + d(x1_n1[i][k], X[k], y1_n1[i][k], Y[k], z1_n1[i][k], Z[k]) * (Y[k] - y1_n1[i][k])
                    DZ_sum = DZ_sum + d(x1_n1[i][k], X[k], y1_n1[i][k], Y[k], z1_n1[i][k], Z[k]) * (Z[k] - z1_n1[i][k])

            dx_1 = (-1)*((w1[i]) * y1_n1[i][k] + z1_n1[i][k]) + D(x1_n1[i][k], X[k], y1_n1[i][k], Y[k], z1_n1[i][k], Z[k]) * (X[k] - x1_n1[i][k])
            dy_1 = (w1[i] * x1_n1[i][k] + a * y1_n1[i][k]) + d_fin_y + D(x1_n1[i][k], X[k], y1_n1[i][k], Y[k], z1_n1[i][k], Z[k]) * (Y[k] - y1_n1[j][k])
            dz_1 = (b + z1_n1[i][k] * (x1_n1[i][k] - c)) + D(x1_n1[i][k], X[k], y1_n1[i][k], Y[k], z1_n1[i][k], Z[k]) * (Z[k] - z1_n1[i][k])
            # dx_1 = (-1)*((w1[i]) * y1_n1[i][k] + z1_n1[i][k]) + DX_sum
            # dy_1 = (w1[i] * x1_n1[i][k] + a * y1_n1[i][k]) + DY_sum + d_fin_y
            # dz_1 = (b + z1_n1[i][k] * (x1_n1[i][k] - c))# + DZ_sum
            # dx_1 = (-1)*((w1[i]) * y1_n1[i][k] + z1_n1[i][k]) + d_fin_x_1 + d_fin_x_2
            # dy_1 = (w1[i] * x1_n1[i][k] + a * y1_n1[i][k])# + d_fin_y
            # dz_1 = (b + z1_n1[i][k] * (x1_n1[i][k] - c))

            # dx_1 = (-1)*((w1[i]) * y1_n1[i][k] + z1_n1[i][k])
            # dy_1 = (w1[i] * x1_n1[i][k] + a * y1_n1[i][k])
            # dz_1 = (b + z1_n1[i][k] * (x1_n1[i][k] - c))

            x1_n1[i][k+1] = x1_n[i][k] + h * dx_1
            y1_n1[i][k+1] = y1_n[i][k] + h * dy_1
            z1_n1[i][k+1] = z1_n[i][k] + h * dz_1

    return x1_n1, y1_n1, z1_n1

def num_rossler(x1_n, y1_n, z1_n, h, a, b, c, w1, d_it, numsteps):
    def d(x1_n, x2_n, y1_n, y2_n):
        d_val = 0
        if ((x1_n - x2_n)**2 + (y1_n - y2_n)**2 < 3**2):
            d_val = 0.2

        return d_val

    # --- connection : d_it = {0.2 or 0}

    # x1_n1 = np.random.rand(5, numsteps)
    # y1_n1 = np.random.rand(5, numsteps)
    # z1_n1 = np.random.rand(5, numsteps)

    dx_1 = ((-w1) * y1_n - z1_n)
    dy_1 = (w1 * x1_n + a * y1_n) # + d(x1_n, y1_n) * (y2_n - y1_n))
    dz_1 = (b + z1_n * (x1_n - c))

    # x1_n1[i] = x1_n[i] + h * dx_1
    # y1_n1[i] = y1_n[i] + h * dy_1
    # z1_n1[i] = z1_n[i] + h * dz_1

    x1_n1 = x1_n + h * dx_1
    y1_n1 = y1_n + h * dy_1
    z1_n1 = z1_n + h * dz_1

    return x1_n1, y1_n1, z1_n1

def num_rossler_freq(x1_n, y1_n, z1_n, h, a, b, c, w1, d_it, numsteps):
    def d(x1_n, x2_n, y1_n, y2_n):
        d_val = 0
        if ((x1_n - x2_n)**2 + (y1_n - y2_n)**2 < 3**2):
            d_val = 0.2
        # d_val = d_it

        return d_val

    # --- connection : d_it = {0.2 or 0}

    dx_1 = ((-w1) * y1_n - z1_n)
    dy_1 = (w1 * x1_n + a * y1_n) # + d(x1_n, x2_n, y1_n, y2_n) * (y2_n - y1_n))
    dz_1 = (b + z1_n * (x1_n - c))

    dx_1_v = ((-w1) * y1_n - z1_n)
    dy_1_v = (w1 * x1_n + a * y1_n)
    dz_1_v = (b + z1_n * (x1_n - c))

    x1_n1 = x1_n + h * dx_1
    y1_n1 = y1_n + h * dy_1
    z1_n1 = z1_n + h * dz_1

    phi_1 = np.arctan(dy_1/dx_1)
    phi_1 = np.arctan(y1_n/x1_n)

    dphi_1 = ((w1 + a) * dx_1 + (-w1 - 1) * dy_1) / (dx_1**2)

    dphi_f_1 = 1 / (1 + (dy_1/dx_1)**2)
    dphi_g_1 = dphi_1
    res_dphi1 = dphi_f_1 * dphi_g_1

    d_dx = (-1)*w1 - 1
    d_dy = w1 + a

    first = dy_1_v * d_dx - dx_1_v * d_dy
    sec = dx_1_v**2 + dy_1_v**2

    res_dphi1 = (first/sec)

    phase_diff_1 = phi_1

    return res_dphi1, phase_diff_1

def rossler_eq(x, y, z, a, b, c, w):
    # Сама система
    dx = (-w) * y - z
    dy = w * x + a * y
    dz = b + z * (x - c)

    phi_1 = np.arctan(dy/dx)

    d_dx = (-1)*w - 1
    d_dy = w + a

    first = dy * d_dx - dx * d_dy
    sec = dx**2 + dy**2

    res_dphi1 = first / sec
    
    return dx, dy, dz, res_dphi1


def solve(x, y, z, phi, v, t, a, b, c, w):
    for i in range(len(t) - 1):
        dx, dy, dz, dphi = rossler_eq(x[i], y[i], z[i], a, b, c, w)

        dt = t[i + 1] - t[i]

        phi[i] = np.arctan(dy / dx)
        v[i] = dphi
    
        x[i + 1] = x[i] + (dx * dt)
        y[i + 1] = y[i] + (dy * dt)
        z[i + 1] = z[i] + (dz * dt)

def visualize_new_attractor(X, Y, Z, t, time_frame, steps_t):
    fig = plt.figure(figsize=(10, 6))  

    col = 'red'
    
    axis_1 = fig.add_subplot(2, 2, 2, projection='3d')
    lorenc_line, = axis_1.plot([], [], [], color=col, linewidth=0.8) # osc

    axis_1_track = fig.add_subplot(2, 2, 4)
    lorenc_line_track, = axis_1_track.plot([], [], color=col, linewidth=0.8) # osc
    lorenc_point_track, = axis_1_track.plot([], [], marker='o', color='black', markersize=3)
    lorenc_point, = axis_1.plot([], [], [], marker='o', color='black', markersize=3)
    
    axis_1.xaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    axis_1.yaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    axis_1.zaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    
    axis_1.set_xlim(-15, 15)
    axis_1.set_ylim(-15, 15)
    axis_1.set_zlim(-25, 35)

    axis_1_track.set_xlim(-22, 22)
    axis_1_track.set_ylim(-22, 22)
    
    axis_1.set_xlabel('X', fontsize=12)
    axis_1.set_ylabel('Y', fontsize=12)
    axis_1.set_zlabel('Z', fontsize=12)

    axis_1_track.set_xlabel('X', fontsize=12)
    axis_1_track.set_ylabel('Y', fontsize=12)
    
    axis_1.set_yticklabels([])
    axis_1.set_xticklabels([])
    axis_1.set_zticklabels([])

    axis_1_track.set_yticklabels([])
    axis_1_track.set_xticklabels([])

    axis_2_x = fig.add_subplot(3, 2, 1)
    lorenc_line3_1_x, = axis_2_x.plot([], [], 'b-', linewidth=0.8) # osc 1
    lorenc_point3_1_x, = axis_2_x.plot([], [], marker='o', color=col, markersize=1)

    axis_2_y = fig.add_subplot(3, 2, 3)
    lorenc_line3_1_y, = axis_2_y.plot([], [], 'b-', linewidth=0.8) # osc 1
    lorenc_point3_1_y, = axis_2_y.plot([], [], marker='o', color=col, markersize=1)

    axis_2_z = fig.add_subplot(3, 2, 5)
    lorenc_line3_1_z, = axis_2_z.plot([], [], 'b-', linewidth=0.8) # osc 1
    lorenc_point3_1_z, = axis_2_z.plot([], [], marker='o', color=col, markersize=1)

    # --- setting the data

    axis_2_x.set_xlim(t[0], t[len(t) - 1])
    axis_2_x.set_xlim(0, time_frame)
    axis_2_x.set_ylim(min(X) - 10, max(X + 10))
    axis_2_x.set_ylabel('x(t)', fontsize=8)

    axis_2_y.set_xlim(t[0], t[len(t) - 1])
    axis_2_y.set_xlim(0, time_frame)
    axis_2_y.set_ylim(min(Y) - 10, max(Y) + 10)
    axis_2_y.set_ylabel('y(t)', fontsize=8)

    axis_2_z.set_xlim(t[0], t[len(t) - 1])
    axis_2_z.set_xlim(0, time_frame)
    axis_2_z.set_ylim(min(Z) - 10, max(Z) + 10)
    axis_2_z.set_ylabel('z(t)', fontsize=8)

    # ---
    
    axis_2_x.set_xlim(t[0], t[len(t) - 1])
    axis_2_x.set_xlim(0, time_frame)
    axis_2_x.set_ylim(min(X) - 10, max(X + 10))
    axis_2_x.set_ylabel('x(t)', fontsize=8)

    axis_2_y.set_xlim(t[0], t[len(t) - 1])
    axis_2_y.set_xlim(0, time_frame)
    axis_2_y.set_ylim(min(Y) - 10, max(Y) + 10)
    axis_2_y.set_ylabel('y(t)', fontsize=8)

    axis_2_z.set_xlim(t[0], t[len(t) - 1])
    axis_2_z.set_xlim(0, time_frame)
    axis_2_z.set_ylim(min(Z) - 10, max(Z) + 10)
    axis_2_z.set_ylabel('z(t)', fontsize=8)

    when_to_track = round((steps_t/100)*98)
    when_to_track = 0

    x1_val_track = np.random.rand(steps_t - when_to_track)
    y1_val_track = np.random.rand(steps_t - when_to_track)
    z1_val_track = np.random.rand(steps_t - when_to_track)

    x1_val_track = X[when_to_track:steps_t]
    y1_val_track = Y[when_to_track:steps_t]
    z1_val_track = Z[when_to_track:steps_t]

    colors = ['violet',
              'aquamarine',
              'gold',
              'darkgreen',
              'maroon',
              'blue']

    def update_all_0(i):
        n = 0
        i = (i) * 10000 - 1
        if (i < steps_t):
# --------------------------- attr 0 ---------------------------
            X_val = X[0:i]
            Y_val = Y[0:i]
            Z_val = Z[0:i]

            tq = t[0:i]
            
            lorenc_line.set_data(X_val, Y_val)
            lorenc_line.set_3d_properties(Z_val)
            lorenc_line.set_color(colors[5])
            lorenc_point.set_3d_properties(Z[i])
            lorenc_line_track.set_color(colors[5])

            #------ x pro
            lorenc_line3_1_x.set_data(tq, X_val)
            lorenc_point3_1_x.set_data(t[i], X[i])
            lorenc_line3_1_x.set_color(colors[5])

            #------ y pro
            lorenc_line3_1_y.set_data(tq, Y_val)
            lorenc_point3_1_y.set_data(t[i], Y[i])
            lorenc_line3_1_y.set_color(colors[5])

            #------ z pro
            lorenc_line3_1_z.set_data(tq, Z_val)
            lorenc_point3_1_z.set_data(t[i], Z[i])
            lorenc_line3_1_z.set_color(colors[5])
        
        # --- from the start
            lorenc_line_track.set_data(X_val, Y_val)
            lorenc_point_track.set_data(X[i], Y[i])
        
        return lorenc_line, lorenc_line_track, lorenc_line3_1_x, lorenc_line3_1_y, lorenc_line3_1_z, lorenc_point, lorenc_point_track, lorenc_point3_1_x, lorenc_point3_1_y, lorenc_point3_1_z

    anim0 = FuncAnimation(fig, update_all_0, frames = np.size(X[0]), interval = 0, blit = True)

    plt.show()
    plt.savefig('rosslerAttractor.png')

    
def visualize(X, Y, Z, x1, y1, z1, t, freq_1, time_frame, steps_t, steps_fr, N):
    fig = plt.figure(figsize=(10, 6))  

    h1 = hilbert(x1[0])
    phi1 = np.unwrap(np.angle(h1))

    col = 'red'
    
    axis_1 = fig.add_subplot(2, 2, 2, projection='3d')
    line1, = axis_1.plot([], [], [], color=col, linewidth=0.8) # osc
    line1_at1, = axis_1.plot([], [], [], color=col, linewidth=0.8) # osc
    line1_at2, = axis_1.plot([], [], [], color=col, linewidth=0.8) # osc
    line1_at3, = axis_1.plot([], [], [], color=col, linewidth=0.8) # osc
    line1_at4, = axis_1.plot([], [], [], color=col, linewidth=0.8) # osc
    lorenc_line, = axis_1.plot([], [], [], color='white', linewidth=0.8) # osc

    axis_1_track = fig.add_subplot(2, 2, 4)
    line1_track, = axis_1_track.plot([], [], color=col, linewidth=0.8) # osc
    line1_track_at1, = axis_1_track.plot([], [], color=col, linewidth=0.8) # osc
    line1_track_at2, = axis_1_track.plot([], [], color=col, linewidth=0.8) # osc
    line1_track_at3, = axis_1_track.plot([], [], color=col, linewidth=0.8) # osc
    line1_track_at4, = axis_1_track.plot([], [], color=col, linewidth=0.8) # osc
    lorenc_line_track, = axis_1_track.plot([], [], color=col, linewidth=0.8) # osc

    point_track1, = axis_1_track.plot([], [], marker='o', color=col, markersize=2)
    point_track1_at1, = axis_1_track.plot([], [], marker='o', color=col, markersize=2)
    point_track1_at2, = axis_1_track.plot([], [], marker='o', color=col, markersize=2)
    point_track1_at3, = axis_1_track.plot([], [], marker='o', color=col, markersize=2)
    point_track1_at4, = axis_1_track.plot([], [], marker='o', color=col, markersize=2)
    lorenc_point_track, = axis_1_track.plot([], [], marker='o', color='black', markersize=3)

    point, = axis_1.plot([], [], [], marker='o', color=col, markersize=2)
    point_at1, = axis_1.plot([], [], [], marker='o', color=col, markersize=2)
    point_at2, = axis_1.plot([], [], [], marker='o', color=col, markersize=2)
    point_at3, = axis_1.plot([], [], [], marker='o', color=col, markersize=2)
    point_at4, = axis_1.plot([], [], [], marker='o', color=col, markersize=2)
    lorenc_point, = axis_1.plot([], [], [], marker='o', color='black', markersize=3)
    
    axis_1.xaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    axis_1.yaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    axis_1.zaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    
    axis_1.set_xlim(-15, 15)
    axis_1.set_ylim(-15, 15)
    axis_1.set_zlim(-25, 35)

    axis_1_track.set_xlim(-22, 22)
    axis_1_track.set_ylim(-22, 22)
    
    axis_1.set_xlabel('X', fontsize=12)
    axis_1.set_ylabel('Y', fontsize=12)
    axis_1.set_zlabel('Z', fontsize=12)

    axis_1_track.set_xlabel('X', fontsize=12)
    axis_1_track.set_ylabel('Y', fontsize=12)
    # axis_1_track.set_ylabel('Z', fontsize=12)
    
    axis_1.set_yticklabels([])
    axis_1.set_xticklabels([])
    axis_1.set_zticklabels([])

    axis_1_track.set_yticklabels([])
    axis_1_track.set_xticklabels([])

    axis_2_x = fig.add_subplot(3, 2, 1)
    line3_1_x, = axis_2_x.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_x_at1, = axis_2_x.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_x_at2, = axis_2_x.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_x_at3, = axis_2_x.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_x_at4, = axis_2_x.plot([], [], 'b-', linewidth=0.8) # osc 1
    lorenc_line3_1_x, = axis_2_x.plot([], [], 'b-', linewidth=0.8) # osc 1
 
    point3_1_x, = axis_2_x.plot([], [], marker='o', color=col, markersize=1)
    point3_1_x_at1, = axis_2_x.plot([], [], marker='o', color=col, markersize=1)
    point3_1_x_at2, = axis_2_x.plot([], [], marker='o', color=col, markersize=1)
    point3_1_x_at3, = axis_2_x.plot([], [], marker='o', color=col, markersize=1)
    point3_1_x_at4, = axis_2_x.plot([], [], marker='o', color=col, markersize=1)
    lorenc_point3_1_x, = axis_2_x.plot([], [], marker='o', color=col, markersize=1)

    axis_2_y = fig.add_subplot(3, 2, 3)
    line3_1_y, = axis_2_y.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_y_at1, = axis_2_y.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_y_at2, = axis_2_y.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_y_at3, = axis_2_y.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_y_at4, = axis_2_y.plot([], [], 'b-', linewidth=0.8) # osc 1
    lorenc_line3_1_y, = axis_2_y.plot([], [], 'b-', linewidth=0.8) # osc 1
 
    point3_1_y, = axis_2_y.plot([], [], marker='o', color=col, markersize=1)
    point3_1_y_at1, = axis_2_y.plot([], [], marker='o', color=col, markersize=1)
    point3_1_y_at2, = axis_2_y.plot([], [], marker='o', color=col, markersize=1)
    point3_1_y_at3, = axis_2_y.plot([], [], marker='o', color=col, markersize=1)
    point3_1_y_at4, = axis_2_y.plot([], [], marker='o', color=col, markersize=1)
    lorenc_point3_1_y, = axis_2_y.plot([], [], marker='o', color=col, markersize=1)

    axis_2_z = fig.add_subplot(3, 2, 5)
    line3_1_z, = axis_2_z.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_z_at1, = axis_2_z.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_z_at2, = axis_2_z.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_z_at3, = axis_2_z.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_1_z_at4, = axis_2_z.plot([], [], 'b-', linewidth=0.8) # osc 1
    lorenc_line3_1_z, = axis_2_z.plot([], [], 'b-', linewidth=0.8) # osc 1
 
    point3_1_z, = axis_2_z.plot([], [], marker='o', color=col, markersize=1)
    point3_1_z_at1, = axis_2_z.plot([], [], marker='o', color=col, markersize=1)
    point3_1_z_at2, = axis_2_z.plot([], [], marker='o', color=col, markersize=1)
    point3_1_z_at3, = axis_2_z.plot([], [], marker='o', color=col, markersize=1)
    point3_1_z_at4, = axis_2_z.plot([], [], marker='o', color=col, markersize=1)
    lorenc_point3_1_z, = axis_2_z.plot([], [], marker='o', color=col, markersize=1)

    # axis_3 = fig.add_subplot(4, 2, 7)
    # line_phi, = axis_3.plot([], [], 'g-', linewidth=0.8)

    # axis_3.set_xlim(0, time_frame)
    # axis_3.set_ylim(min(phi1), max(phi1))
    # axis_3.set_ylabel('phi(t) - фаза', fontsize=8)
    # point_phi, = axis_3.plot([], [], marker='o', color='b', markersize=1)

    # --- setting the data

    axis_2_x.set_xlim(t[0], t[len(t) - 1])
    axis_2_x.set_xlim(0, time_frame)
    axis_2_x.set_ylim(min(x1[0]) - 10, max(x1[0] + 10))
    axis_2_x.set_ylabel('x(t)', fontsize=8)

    axis_2_y.set_xlim(t[0], t[len(t) - 1])
    axis_2_y.set_xlim(0, time_frame)
    axis_2_y.set_ylim(min(y1[0]) - 10, max(y1[0]) + 10)
    axis_2_y.set_ylabel('y(t)', fontsize=8)

    axis_2_z.set_xlim(t[0], t[len(t) - 1])
    axis_2_z.set_xlim(0, time_frame)
    axis_2_z.set_ylim(min(z1[0]) - 10, max(z1[0]) + 10)
    axis_2_z.set_ylabel('z(t)', fontsize=8)

    # ---
    
    axis_2_x.set_xlim(t[0], t[len(t) - 1])
    axis_2_x.set_xlim(0, time_frame)
    axis_2_x.set_ylim(min(x1[0]) - 10, max(x1[0] + 10))
    axis_2_x.set_ylabel('x(t)', fontsize=8)

    axis_2_y.set_xlim(t[0], t[len(t) - 1])
    axis_2_y.set_xlim(0, time_frame)
    axis_2_y.set_ylim(min(y1[0]) - 10, max(y1[0]) + 10)
    axis_2_y.set_ylabel('y(t)', fontsize=8)

    axis_2_z.set_xlim(t[0], t[len(t) - 1])
    axis_2_z.set_xlim(0, time_frame)
    axis_2_z.set_ylim(min(z1[0]) - 10, max(z1[0]) + 10)
    axis_2_z.set_ylabel('z(t)', fontsize=8)

    when_to_track = round((steps_t/200)*50)
    when_to_track = 0

    x1_val_track = np.random.rand(5, steps_t - when_to_track)
    y1_val_track = np.random.rand(5, steps_t - when_to_track)
    z1_val_track = np.random.rand(5, steps_t - when_to_track)

    X_val_track = np.random.rand(steps_t - when_to_track)
    Y_val_track = np.random.rand(steps_t - when_to_track)
    Z_val_track = np.random.rand(steps_t - when_to_track)

    for n in range (5):
        x1_val_track[n] = x1[n][when_to_track:steps_t]
        y1_val_track[n] = y1[n][when_to_track:steps_t]
        z1_val_track[n] = z1[n][when_to_track:steps_t]

    X_val_track = X[when_to_track:steps_t]
    Y_val_track = Y[when_to_track:steps_t]
    Z_val_track = Z[when_to_track:steps_t]

    colors = ['violet',
              'aquamarine',
              'gold',
              'darkgreen',
              'maroon',
              'black']
    
    colors = ['violet',
              'violet',
              'violet',
              'violet',
              'violet',
              'black']

    def update_all_0(i):
        n = 0
        i = (i) * 2000000 - 1
        if (i < steps_t):
# --------------------------- attr 0 ---------------------------
            x1_val = x1[n][0:i]
            y1_val = y1[n][0:i]
            z1_val = z1[n][0:i]

            X_val = X[0:i]
            Y_val = Y[0:i]
            Z_val = Z[0:i]

            _x1_val = x1[n][when_to_track:i]
            _y1_val = y1[n][when_to_track:i]
            _z1_val = z1[n][when_to_track:i]

            _X_val = X[when_to_track:i]
            _Y_val = Y[when_to_track:i]
            _Z_val = Z[when_to_track:i]

            tq = t[0:i]

            v1_val = freq_1[n][0:i]
            
            line1.set_data(_x1_val, _y1_val)
            line1.set_3d_properties(_z1_val)
            line1.set_color(colors[n])
            # point.set_3d_properties(z1[n][i])

            # lorenc_line.set_data(X_val, Y_val)
            # lorenc_line.set_3d_properties(Z_val)
            # lorenc_point.set_3d_properties(Z[i])

            line1_track.set_color(colors[n])
            lorenc_line_track.set_color(colors[5])

            lorenc_line.set_data(_X_val, _Y_val)
            lorenc_line.set_3d_properties(_Z_val)
            lorenc_line.set_color(colors[5])
            # lorenc_point.set_data(X[i], Y[i])
            # lorenc_point.set_3d_properties(Z[i])

            #------ x pro

            line3_1_x.set_data(tq, x1_val)
            point3_1_x.set_data(t[i], x1[n][i])
            line3_1_x.set_color(colors[n])

            lorenc_line3_1_x.set_data(tq, X_val)
            lorenc_point3_1_x.set_data(t[i], X[i])
            lorenc_line3_1_x.set_color(colors[5])

            #------ y pro

            line3_1_y.set_data(tq, y1_val)
            point3_1_y.set_data(t[i], y1[n][i])
            line3_1_y.set_color(colors[n])

            lorenc_line3_1_y.set_data(tq, Y_val)
            lorenc_point3_1_y.set_data(t[i], Y[i])
            lorenc_line3_1_y.set_color(colors[5])

            #------ z pro

            line3_1_z.set_data(tq, z1_val)
            point3_1_z.set_data(t[i], z1[n][i])
            line3_1_z.set_color(colors[n])

            lorenc_line3_1_z.set_data(tq, Z_val)
            lorenc_point3_1_z.set_data(t[i], Z[i])
            lorenc_line3_1_z.set_color(colors[5])
        
        # --- from the start
            # line1_track.set_data(x1_val, y1_val)
            # point_track1.set_data(x1[n][i], y1[n][i])

            # lorenc_line_track.set_data(X_val, Y_val)
            # lorenc_point_track.set_data(X[i], Y[i])

            line1_track.set_data(_x1_val, _y1_val)
            if (i > when_to_track):
                point_track1.set_data(x1[n][i], y1[n][i])
            lorenc_line_track.set_data(_X_val, _Y_val)
            # lorenc_point_track.set_data(X[i], Y[i])
        
            # if (i > when_to_track):
                # line1_track.set_data(x1_val_track, y1_val_track)
                # point_track1.set_data(x1[n][i], y1[n][i])
                # lorenc_line_track.set_data(X_val_track, Y_val_track)
                # lorenc_point_track.set_data(X[i], Y[i])

# --------------------------- attr 1 ---------------------------
            n = 1
            x1_val = x1[n][0:i]
            y1_val = y1[n][0:i]
            z1_val = z1[n][0:i]

            _x1_val = x1[n][when_to_track:i]
            _y1_val = y1[n][when_to_track:i]
            _z1_val = z1[n][when_to_track:i]

            tq = t[0:i]

            v1_val = freq_1[n][0:i]
            
            line1_at1.set_data(_x1_val, _y1_val)
            line1_at1.set_3d_properties(_z1_val)
            # point_at1.set_3d_properties(z1[n][i])
            line1_at1.set_color(colors[n])
            if (i > when_to_track):
                point_at1.set_color(colors[n])
            else:
                point_at1.set_color('white')

            line1_track_at1.set_color(colors[n])

            #------ x pro

            line3_1_x_at1.set_data(tq, x1_val)
            point3_1_x_at1.set_data(t[i], x1[n][i])
            line3_1_x_at1.set_color(colors[n])

            #------ y pro

            line3_1_y_at1.set_data(tq, y1_val)
            point3_1_y_at1.set_data(t[i], y1[n][i])
            line3_1_y_at1.set_color(colors[n])

            #------ z pro

            line3_1_z_at1.set_data(tq, z1_val)
            point3_1_z_at1.set_data(t[i], z1[n][i])
            line3_1_z_at1.set_color(colors[n])

            line1_track_at1.set_data(_x1_val, _y1_val)
            if (i > when_to_track):
                point_track1_at1.set_data(x1[n][i], y1[n][i])

            # if (i > when_to_track):
            #     line1_track_at1.set_data(x1_val_track, y1_val_track)
            #     point_track1_at1.set_data(x1[n][i], y1[n][i])

# --------------------------- attr 2 ---------------------------
            n = 2
            x1_val = x1[n][0:i]
            y1_val = y1[n][0:i]
            z1_val = z1[n][0:i]

            _x1_val = x1[n][when_to_track:i]
            _y1_val = y1[n][when_to_track:i]
            _z1_val = z1[n][when_to_track:i]

            tq = t[0:i]

            v1_val = freq_1[n][0:i]
            
            line1_at2.set_data(_x1_val, _y1_val)
            line1_at2.set_3d_properties(_z1_val)

            # point_at2.set_3d_properties(z1[n][i])
            line1_at2.set_color(colors[n])
            line1_track_at2.set_color(colors[n])

            #------ x pro

            line3_1_x_at2.set_data(tq, x1_val)
            point3_1_x_at2.set_data(t[i], x1[n][i])
            line3_1_x_at2.set_color(colors[n])

            #------ y pro

            line3_1_y_at2.set_data(tq, y1_val)
            point3_1_y_at2.set_data(t[i], y1[n][i])
            line3_1_y_at2.set_color(colors[n])

            #------ z pro

            line3_1_z_at2.set_data(tq, z1_val)
            point3_1_z_at2.set_data(t[i], z1[n][i])
            line3_1_z_at2.set_color(colors[n])

            line1_track_at2.set_data(_x1_val, _y1_val)
            if (i > when_to_track):
                point_track1_at2.set_data(x1[n][i], y1[n][i])

            # if (i > when_to_track):
            #     line1_track_at2.set_data(x1_val_track, y1_val_track)
            #     point_track1_at2.set_data(x1[n][i], y1[n][i])

# --------------------------- attr 3 ---------------------------
            n = 3
            x1_val = x1[n][0:i]
            y1_val = y1[n][0:i]
            z1_val = z1[n][0:i]

            _x1_val = x1[n][when_to_track:i]
            _y1_val = y1[n][when_to_track:i]
            _z1_val = z1[n][when_to_track:i]

            tq = t[0:i]

            v1_val = freq_1[n][0:i]
            
            line1_at3.set_data(_x1_val, _y1_val)
            line1_at3.set_3d_properties(_z1_val)

            # point_at3.set_3d_properties(z1[n][i])
            line1_at3.set_color(colors[n])
            if (i > when_to_track):
                point_at3.set_color(colors[n])
            else:
                point_at3.set_color('white')

            line1_track_at3.set_color(colors[n])

            #------ x pro

            line3_1_x_at3.set_data(tq, x1_val)
            point3_1_x_at3.set_data(t[i], x1[n][i])
            line3_1_x_at3.set_color(colors[n])

            #------ y pro

            line3_1_y_at3.set_data(tq, y1_val)
            point3_1_y_at3.set_data(t[i], y1[n][i])
            line3_1_y_at3.set_color(colors[n])

            #------ z pro

            line3_1_z_at3.set_data(tq, z1_val)
            point3_1_z_at3.set_data(t[i], z1[n][i])
            line3_1_z_at3.set_color(colors[n])

            line1_track_at3.set_data(_x1_val, _y1_val)
            if (i > when_to_track):
                point_track1_at3.set_data(x1[n][i], y1[n][i])

            # if (i > when_to_track):
            #     line1_track_at3.set_data(x1_val_track, y1_val_track)
            #     point_track1_at3.set_data(x1[n][i], y1[n][i])

# --------------------------- attr 4 ---------------------------
            n = 4
            x1_val = x1[n][0:i]
            y1_val = y1[n][0:i]
            z1_val = z1[n][0:i]

            _x1_val = x1[n][when_to_track:i]
            _y1_val = y1[n][when_to_track:i]
            _z1_val = z1[n][when_to_track:i]

            tq = t[0:i]

            v1_val = freq_1[n][0:i]
            
            line1_at4.set_data(_x1_val, _y1_val)
            line1_at4.set_3d_properties(_z1_val)

            # point_at4.set_3d_properties(z1[n][i])
            line1_at4.set_color(colors[n])
            line1_track_at4.set_color(colors[n])

            #------ x pro

            line3_1_x_at4.set_data(tq, x1_val)
            point3_1_x_at4.set_data(t[i], x1[n][i])
            line3_1_x_at4.set_color(colors[n])

            #------ y pro

            line3_1_y_at4.set_data(tq, y1_val)
            point3_1_y_at4.set_data(t[i], y1[n][i])
            line3_1_y_at4.set_color(colors[n])

            #------ z pro

            line3_1_z_at4.set_data(tq, z1_val)
            point3_1_z_at4.set_data(t[i], z1[n][i])
            line3_1_z_at4.set_color(colors[n])

            line1_track_at4.set_data(_x1_val, _y1_val)
            if (i > when_to_track):
                point_track1_at4.set_data(x1[n][i], y1[n][i])

            # if (i > when_to_track):
            #     line1_track_at4.set_data(x1_val_track, y1_val_track)
            #     point_track1_at4.set_data(x1[n][i], y1[n][i])

        # return line1, line1_track, line3_1_x, line3_1_y, line3_1_z, point, point_track1, point3_1_x, point3_1_y, point3_1_z, line1_at1, line1_track_at1, line3_1_x_at1, line3_1_y_at1, line3_1_z_at1, point_at1, point_track1_at1, point3_1_x_at1, point3_1_y_at1, point3_1_z_at1, line1_at2, line1_track_at2, line3_1_x_at2, line3_1_y_at2, line3_1_z_at2, point_at2, point_track1_at2, point3_1_x_at2, point3_1_y_at2, point3_1_z_at2, line1_at3, line1_track_at3, line3_1_x_at3, line3_1_y_at3, line3_1_z_at3, point_at3, point_track1_at3, point3_1_x_at3, point3_1_y_at3, point3_1_z_at3, line1_at4, line1_track_at4, line3_1_x_at4, line3_1_y_at4, line3_1_z_at4, point_at4, point_track1_at4, point3_1_x_at4, point3_1_y_at4, point3_1_z_at4
        return lorenc_line, line1, lorenc_line_track, line1_track, lorenc_line3_1_x, line3_1_x, lorenc_line3_1_y, line3_1_y, lorenc_line3_1_z, line3_1_z, point, lorenc_point, lorenc_point_track, point_track1, lorenc_point3_1_x, lorenc_point3_1_y, lorenc_point3_1_z, point3_1_x, point3_1_y, point3_1_z, line1_at1, line1_track_at1, line3_1_x_at1, line3_1_y_at1, line3_1_z_at1, point_at1, point_track1_at1, point3_1_x_at1, point3_1_y_at1, point3_1_z_at1, line1_at2, line1_track_at2, line3_1_x_at2, line3_1_y_at2, line3_1_z_at2, point_at2, point_track1_at2, point3_1_x_at2, point3_1_y_at2, point3_1_z_at2, line1_at3, line1_track_at3, line3_1_x_at3, line3_1_y_at3, line3_1_z_at3, point_at3, point_track1_at3, point3_1_x_at3, point3_1_y_at3, point3_1_z_at3, line1_at4, line1_track_at4, line3_1_x_at4, line3_1_y_at4, line3_1_z_at4, point_at4, point_track1_at4, point3_1_x_at4, point3_1_y_at4, point3_1_z_at4

    # def update_phi(j):
    #     j = (j - 1) * 100 - 1

    #     if (j < steps_t & j < steps_fr):
    #         tq = t[0:j]

    #         phi_val = phi1[0:j]

    #         line_phi.set_data(tq, phi_val)
    #         point_phi.set_data(t[j], phi1[j])
    #         line_phi.set_color("blue")
        
    #     return line_phi, point_phi
    
    # anim_phi = FuncAnimation(fig, update_phi, frames = np.size(x1[0]), interval = 0, blit =True)

    anim0 = FuncAnimation(fig, update_all_0, frames = np.size(x1[0]), interval = 0, blit = True)
    # anim1 = FuncAnimation(fig, update_all_1, frames = np.size(x1[1]), interval = 0, blit = True)
    # anim2 = FuncAnimation(fig, update_all_2, frames = np.size(x1[2]), interval = 0, blit = True)
    # anim3 = FuncAnimation(fig, update_all_3, frames = np.size(x1[3]), interval = 0, blit = True)
    # anim4 = FuncAnimation(fig, update_all_4, frames = np.size(x1[4]), interval = 0, blit = True)

    plt.show()
    plt.savefig('rosslerAttractor.png')


def main():
    N_of_agents = 5

    a = 0.22
    b = 0.1
    c = 8.5

    t_ini = 0
    t_fin = 200
    h = 0.0001
    numsteps = int((t_fin - t_ini) / h)
    t = np.linspace(t_ini, t_fin, numsteps)

    d_ini = 0
    d_fin = 0.32
    d_h = 0.00000016
    d_numsteps = int((d_fin - d_ini) / d_h)
    d = np.linspace(d_ini, d_fin, d_numsteps)

    x1 = np.random.rand(N_of_agents, numsteps)
    y1 = np.random.rand(N_of_agents, numsteps)
    z1 = np.random.rand(N_of_agents, numsteps)

    X = np.random.rand(numsteps)
    Y = np.random.rand(numsteps)
    Z = np.random.rand(numsteps)

    w1 = np.random.rand(N_of_agents)

    freq_1 = np.random.rand(N_of_agents, numsteps)
    phase_diff_1 = np.random.rand(N_of_agents, numsteps)

    for i in range(N_of_agents):
        # w1[i]=np.random.uniform(0.93,1.07) # random in the range of (0,93 .. 1,07)
        w1[0] = 0.9809
        w1[1] = 0.9808
        w1[2] = 0.9805
        w1[3] = 0.9803
        w1[4] = 0.9801

        # w1[0] = 0.95
        # w1[1] = 1
        # w1[2] = 1.07
        # w1[3] = 0.98
        # w1[4] = 1.02

        # w1[0] = 1.021
        # w1[1] = 1.022
        # w1[2] = 1.023
        # w1[3] = 1.024
        # w1[4] = 1.025

        x1[i] = np.zeros(numsteps)
        y1[i] = np.zeros(numsteps)
        z1[i] = np.zeros(numsteps)
        freq_1[i] = np.zeros(numsteps)

        phase_diff_1[i] = np.zeros(numsteps)

        # # random start point
        x1[i][0] = np.random.uniform(-10, 10)
        y1[i][0] = np.random.uniform(-10, 10)
        z1[i][0] = np.random.uniform(-10, 10)
        # x1[i][0] = 0
        # y1[i][0] = 0
        # z1[i][0] = 0

        X[0] = 2
        Y[0] = 1
        Z[0] = 0

        x1[0][0] = -1
        y1[0][0] = 2
        z1[0][0] = 6

        x1[1][0] = 5
        y1[1][0] = -3
        z1[1][0] = 2

        x1[2][0] = 0
        y1[2][0] = -5
        z1[2][0] = 23

        x1[3][0] = 2
        y1[3][0] = -2
        z1[3][0] = -12

        x1[4][0] = -8
        y1[4][0] = 4
        z1[4][0] = 14

        # x1[0]=3
        # y1[0]=12
        # z1[0]=5

        # x1[1]=7
        # y1[1]=-3
        # z1[1]=-15

        # x1[2]=6
        # y1[2]=6
        # z1[2]=10

        # x1[3]=0
        # y1[3]=0
        # z1[3]=0

        # x1[4]=-5
        # y1[4]=-1
        # z1[4]=5

        # for k in range(numsteps-1):
        #     [x1[i][k+1], y1[i][k+1], z1[i][k+1]] = num_rossler(x1[i][k], y1[i][k], z1[i][k], t[k+1] - t[k], a, b, c, w1, d[k], numsteps)

        # for k in range(d_numsteps-1):
        #     [freq_1[i][k+1], phase_diff_1[i][k+1]] = num_rossler_freq(x1[i][k], y1[i][k], z1[i][k], t[k+1]-t[k], a, b, c, w1[i], d[k], numsteps)

    X, Y, Z = num_lorenc(X, Y, Z, h, 10, 28, 8/3, numsteps)
    x1, y1, z1 = num_rossler_together(x1, y1, z1, h, a, b, c, w1, d, numsteps, X, Y, Z)

    visualize(X, Y, Z, x1, y1, z1, t, freq_1, t_fin, numsteps, d_numsteps, N_of_agents)
    # visualize_new_attractor(X, Y, Z, t, t_fin, numsteps)

    
if __name__ == "__main__":
    main()
