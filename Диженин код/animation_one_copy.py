import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import hilbert

import math

def num_lorenc(X1_n, Y1_n, Z1_n, X2_n, Y2_n, Z2_n, h, sigma, ro, Beta, numsteps):
    x1_n1 = X1_n
    y1_n1 = Y1_n
    z1_n1 = Z1_n

    dx_1 = sigma * (y1_n1 - x1_n1)
    dy_1 = x1_n1 * (ro - z1_n1) - y1_n1
    dz_1 = x1_n1 * y1_n1 - Beta * z1_n1

    x1_n1=X1_n+h*dx_1
    y1_n1=Y1_n+h*dy_1
    z1_n1=Z1_n+h*dz_1

    return x1_n1, y1_n1, z1_n1

def num_rossler(x1_n, y1_n, z1_n, x2_n, y2_n, z2_n, h, a, b, c, w1, w2, d_it):
    def d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n):
        d_val = 0
        if ((x1_n - x2_n)**2 + (y1_n - y2_n)**2 + (z1_n - z2_n)**2 < 3**2):
        # if ((x1_n - x2_n)**2 + (y1_n - y2_n)**2 < 3**2):
            d_val = 0.2

        return d_val

    # --- connection : d_it = {0.2 or 0}

    dx_1=(-w1*y1_n-z1_n + d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n) * (x2_n - x1_n))
    # dx_1=(-w1*y1_n-z1_n + d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n) * (x2_n - x1_n) + d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n) / (x1_n - x2_n))
    dy_1=(w1*x1_n+a*y1_n)
    dz_1=(b+z1_n*(x1_n-c))

    dx_2=(-w2*y2_n-z2_n + d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n) * (x1_n - x2_n))
    # dx_2=(-w2*y2_n-z2_n + d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n) * (x2_n - x1_n) + d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n) / (x1_n - x2_n))
    dy_2=(w2*x2_n+a*y2_n)
    dz_2=(b+z2_n*(x2_n-c))


    # --- connection : d_it >= 0

    # dx_1=(-w1*y1_n-z1_n)
    # dy_1=(w1*x1_n+a*y1_n + d_it * (y2_n - y1_n))
    # dz_1=(b+z1_n*(x1_n-c))

    # dx_2=(-w2*y2_n-z2_n)
    # dy_2=(w2*x2_n+a*y2_n + d_it * (y1_n - y2_n))
    # dz_2=(b+z2_n*(x2_n-c))

    # --- no connection

    # dx_1_v=(-w1*y1_n-z1_n)
    # dy_1_v=(w1*x1_n+a*y1_n)
    # dz_1_v=(b+z1_n*(x1_n-c))

    # dx_2_v=(-w2*y2_n-z2_n)
    # dy_2_v=(w2*x2_n+a*y2_n)
    # dz_2_v=(b+z2_n*(x2_n-c))

    # dx_1=(-w1*y1_n-z1_n)
    # dy_1=(w1*x1_n+a*y1_n + d_it * (y2_n - y1_n))
    # dz_1=(b+z1_n*(x1_n-c))

    # dx_2=(-w2*y2_n-z2_n)
    # dy_2=(w2*x2_n+a*y2_n + d_it * (y1_n - y2_n))
    # dz_2=(b+z2_n*(x2_n-c))

    x1_n1=x1_n+h*dx_1
    y1_n1=y1_n+h*dy_1
    z1_n1=z1_n+h*dz_1

    x2_n1=x2_n+h*dx_2
    y2_n1=y2_n+h*dy_2
    z2_n1=z2_n+h*dz_2

    phi_1 = np.arctan(dy_1/dx_1)
    phi_1 = np.arctan(y1_n/x1_n)

    dphi_1 = ((w1 + a) * dx_1 + (-w1 - 1) * dy_1) / (dx_1**2)

    phi_2 = np.arctan(dy_2/dx_2)
    phi_2 = np.arctan(y2_n/x2_n)
    dphi_2 = ((w2 + a) * dx_2 + (-w2 - 1) * dy_2) / (dx_2**2)

    dphi_f_1 = 1 / (1 + (dy_1/dx_1)**2)
    dphi_g_1 = dphi_1
    res_dphi1 = dphi_f_1 * dphi_g_1

    dphi_f_2 = 1 / (1 + (dy_2/dx_2)**2)
    dphi_g_2 = dphi_2
    res_dphi2 = dphi_f_2 * dphi_g_2

    d_dx = (-1)*w1 - 1
    d_dy = w1 + a

    first = dy_1 * d_dx - dx_1 * d_dy
    sec = dx_1**2 + dy_1**2

    d_dx = (-1)*w2 - 1
    d_dy = w2 + a

    first = dy_2 * d_dx - dx_2 * d_dy
    sec = dx_2**2 + dy_2**2

    phase_diff_1 = phi_1

    return x1_n1, y1_n1, z1_n1, x2_n1, y2_n1, z2_n1

def num_rossler_freq(x1_n, y1_n, z1_n, x2_n, y2_n, z2_n, h, a, b, c, w1, w2, d_it):
    def d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n):
        d_val = 0
        if ((x1_n - x2_n)**2 + (y1_n - y2_n)**2 + (z1_n - z2_n)**2 < 3**2):
            d_val = 0.2

        return d_val

    # --- connection : d_it = {0.2 or 0}

    dx_1=(-w1*y1_n-z1_n)
    dy_1=(w1*x1_n+a*y1_n + d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n) * (y2_n - y1_n))
    dz_1=(b+z1_n*(x1_n-c))

    dx_2=(-w2*y2_n-z2_n)
    dy_2=(w2*x2_n+a*y2_n + d(x1_n, x2_n, y1_n, y2_n, z1_n, z2_n) * (y1_n - y2_n))
    dz_2=(b+z2_n*(x2_n-c))


    # --- connection : d_it >= 0

    # dx_1=(-w1*y1_n-z1_n)
    # dy_1=(w1*x1_n+a*y1_n + d_it * (y2_n - y1_n))
    # dz_1=(b+z1_n*(x1_n-c))

    # dx_2=(-w2*y2_n-z2_n)
    # dy_2=(w2*x2_n+a*y2_n + d_it * (y1_n - y2_n))
    # dz_2=(b+z2_n*(x2_n-c))

    # --- no connection

    dx_1_v=(-w1*y1_n-z1_n)
    dy_1_v=(w1*x1_n+a*y1_n)
    dz_1_v=(b+z1_n*(x1_n-c))

    dx_2_v=(-w2*y2_n-z2_n)
    dy_2_v=(w2*x2_n+a*y2_n)
    dz_2_v=(b+z2_n*(x2_n-c))

    x1_n1=x1_n+h*dx_1
    y1_n1=y1_n+h*dy_1
    z1_n1=z1_n+h*dz_1

    x2_n1=x2_n+h*dx_2
    y2_n1=y2_n+h*dy_2
    z2_n1=z2_n+h*dz_2

    phi_1 = np.arctan(dy_1/dx_1)
    phi_1 = np.arctan(y1_n/x1_n)

    dphi_1 = ((w1 + a) * dx_1 + (-w1 - 1) * dy_1) / (dx_1**2)

    phi_2 = np.arctan(dy_2/dx_2)
    phi_2 = np.arctan(y2_n/x2_n)
    dphi_2 = ((w2 + a) * dx_2 + (-w2 - 1) * dy_2) / (dx_2**2)

    dphi_f_1 = 1 / (1 + (dy_1/dx_1)**2)
    dphi_g_1 = dphi_1
    res_dphi1 = dphi_f_1 * dphi_g_1

    dphi_f_2 = 1 / (1 + (dy_2/dx_2)**2)
    dphi_g_2 = dphi_2
    res_dphi2 = dphi_f_2 * dphi_g_2

    d_dx = (-1)*w1 - 1
    d_dy = w1 + a

    first = dy_1_v * d_dx - dx_1_v * d_dy
    sec = dx_1_v**2 + dy_1_v**2

    res_dphi1 = (first/sec)

    d_dx = (-1)*w2 - 1
    d_dy = w2 + a

    first = dy_2_v * d_dx - dx_2_v * d_dy
    sec = dx_2_v**2 + dy_2_v**2

    res_dphi2 = first/sec

    phase_diff_1 = phi_1

    return res_dphi1, res_dphi2, phase_diff_1


def rossler_eq(x, y, z, a, b, c, w):
    # Сама система
    dx = (-w) * y - z
    dy = w * x + a * y
    dz = b + z * (x - c)

    phi_1 = np.arctan(dy/dx)

    d_dx = (-1)*w - 1
    d_dy = w + a

    # dphi_1 = (w * dx * (z - w * y) + dy * (a * z - w * a + (w**2) * x + w * a * y) + dz * (w * x + a * y)) / ((w**2) * (y**2) + 2 * w * y * z + (z**2))
    # dphi_1 = (d_dx * dy - dx * d_dy) / (dy ** 2)

    first = dy * d_dx - dx * d_dy
    sec = dx**2 + dy**2

    # Производная сложной функции
    # dphi_f_1 = 1 / (1 + (dy/dx)**2)
    # dphi_g_1 = dphi_1
    # #  -->
    # res_dphi1 = dphi_f_1 * dphi_g_1
    res_dphi1 = first/sec
    
    return dx, dy, dz, res_dphi1


def solve(x, y, z, phi, v, t, a, b, c, w):
    for i in range(len(t) - 1):
        dx, dy, dz, dphi = rossler_eq(x[i], y[i], z[i], a, b, c, w)

        dt = t[i + 1] - t[i]

        phi[i] = np.arctan(dy/dx)
        v[i] = dphi
    
        x[i + 1] = x[i] + (dx * dt)
        y[i + 1] = y[i] + (dy * dt)
        z[i + 1] = z[i] + (dz * dt)

def get_val(out, vec, start_point, end_point):
    for j in range(end_point - start_point - 1):
        for i in range(end_point - 1):
            if (i > start_point & i < end_point):
                out[j] = vec[i]
            i = (i - 1) * 10000 - 1
        j = (j - 1) * 10000 - 1
    return out
        
def visualize(x1, y1, z1, x2, y2, z2, t, freq_1, freq_2, time_frame, steps_t, steps_fr):
    fig = plt.figure(figsize=(10, 6))  

    h1 = hilbert(x1)
    phi1 = np.unwrap(np.angle(h1))

    h2 = hilbert(x2)
    phi2 = np.unwrap(np.angle(h2))

    x_max = max(max(x1), max(x2))
    y_max = max(max(y1), max(y2))
    z_max = max(max(z1), max(z2))

    x_min = min(min(x1), min(x2))
    y_min = min(min(y1), min(y2))
    z_min = min(min(z1), min(z2))

    phi_max = max(max(phi1), max(phi2))
    phi_min = min(min(phi1), min(phi2))

    v_max = max(max(freq_1), max(freq_2))
    v_min = min(min(freq_1), min(freq_2))
    
    axis_1 = fig.add_subplot(2, 2, 2, projection='3d')
    line1, = axis_1.plot([], [], [], color='b', linewidth=0.8) # osc
    line2, = axis_1.plot([], [], [], color='b', linewidth=0.8) # osc

    axis_1_track = fig.add_subplot(2, 2, 4)
    line1_track, = axis_1_track.plot([], [], color='b', linewidth=0.8) # osc
    line2_track, = axis_1_track.plot([], [], color='b', linewidth=0.8) # osc

    point_track1, = axis_1_track.plot([], [], marker='o', color='r', markersize=2)
    point_track2, = axis_1_track.plot([], [], marker='o', color='b', markersize=2)
    
    point, = axis_1.plot([], [], [], marker='o', color='b', markersize=1)
    
    axis_1.xaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    axis_1.yaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    axis_1.zaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    
    axis_1.set_xlim(-28, 28)
    axis_1.set_ylim(-28, 28)
    axis_1.set_zlim(0, 50)

    axis_1_track.set_xlim(-28, 28)
    axis_1_track.set_ylim(-28, 28)

    # axis_1.set_xlim(-10, 10)
    # axis_1.set_ylim(-10, 10)
    # axis_1.set_zlim(0, 1)
    
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

    axis_2_x = fig.add_subplot(4, 2, 1)
    line3_1_x, = axis_2_x.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_2_x, = axis_2_x.plot([], [], 'b-', linewidth=0.8) # osc 2
 
    point3_1_x, = axis_2_x.plot([], [], marker='o', color='r', markersize=1)
    point3_2_x, = axis_2_x.plot([], [], marker='o', color='b', markersize=1)

    axis_2_y = fig.add_subplot(4, 2, 3)
    line3_1_y, = axis_2_y.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_2_y, = axis_2_y.plot([], [], 'b-', linewidth=0.8) # osc 2
 
    point3_1_y, = axis_2_y.plot([], [], marker='o', color='r', markersize=1)
    point3_2_y, = axis_2_y.plot([], [], marker='o', color='b', markersize=1)

    axis_2_z = fig.add_subplot(4, 2, 5)
    line3_1_z, = axis_2_z.plot([], [], 'b-', linewidth=0.8) # osc 1
    line3_2_z, = axis_2_z.plot([], [], 'b-', linewidth=0.8) # osc 2
 
    point3_1_z, = axis_2_z.plot([], [], marker='o', color='r', markersize=1)
    point3_2_z, = axis_2_z.plot([], [], marker='o', color='b', markersize=1)

    axis_2_x.set_xlim(t[0], t[len(t) - 1])
    axis_2_x.set_ylim(x_min, x_max)
    axis_2_x.set_xlim(0, time_frame)
    axis_2_x.set_ylabel('x(t)', fontsize=8)

    axis_2_y.set_xlim(t[0], t[len(t) - 1])
    axis_2_y.set_ylim(y_min, y_max)
    axis_2_y.set_xlim(0, time_frame)
    axis_2_y.set_ylabel('y(t)', fontsize=8)

    axis_2_z.set_xlim(t[0], t[len(t) - 1])
    axis_2_z.set_ylim(z_min, z_max)
    axis_2_z.set_xlim(0, time_frame)
    axis_2_z.set_ylabel('z(t)', fontsize=8)

    axis_3 = fig.add_subplot(4, 2, 7)
    line_phi, = axis_3.plot([], [], 'g-', linewidth=0.8)

    axis_3.set_xlim(0, time_frame)
    axis_3.set_ylim(phi_min, phi_max)
    axis_3.set_ylabel('phi(t) - фаза', fontsize=8)
    point_phi, = axis_3.plot([], [], marker='o', color='b', markersize=1)

    # axis_4 = fig.add_subplot(4, 2, 9)
    # axis_4.set_ylabel('v(t) - частота', fontsize=8)
    # axis_4.set_ylim(v_min, 200)
    # axis_4.set_xlim(0, time_frame)
    # line_v1, = axis_4.plot([], [], 'g-', linewidth=0.8)
    # line_v2, = axis_4.plot([], [], 'g-', linewidth=0.8)
    
    # point_v1, = axis_4.plot([], [], marker='o', color='r', markersize=4)
    # point_v2, = axis_4.plot([], [], marker='o', color='b', markersize=4)

    when_to_track = round((steps_t/20)*18)
    # when_to_track = 0

    x1_val_track=np.zeros(steps_t-when_to_track)
    x2_val_track=np.zeros(steps_t-when_to_track)
    y1_val_track=np.zeros(steps_t-when_to_track)
    y2_val_track=np.zeros(steps_t-when_to_track)

    z1_val_track=np.zeros(steps_t-when_to_track)
    z2_val_track=np.zeros(steps_t-when_to_track)

    x1_val_track=x1[when_to_track:steps_t]
    x2_val_track=x2[when_to_track:steps_t]
    y1_val_track=y1[when_to_track:steps_t]
    y2_val_track=y2[when_to_track:steps_t]

    z1_val_track=z1[when_to_track:steps_t]
    z2_val_track=z2[when_to_track:steps_t]

    # x1_val_track=get_val(x1_val_track, x1, when_to_track, steps_t)
    # x2_val_track=get_val(x2_val_track, x2, when_to_track, steps_t)
    # y1_val_track=get_val(y1_val_track, y1, when_to_track, steps_t)
    # y2_val_track=get_val(y2_val_track, y2, when_to_track, steps_t)
    
    def update_all(i):
        i = (i - 1) * 200000 - 1

        if (i < steps_t):
            x1_val = x1[0:i]
            y1_val = y1[0:i]
            z1_val = z1[0:i]

            x2_val = x2[0:i]
            y2_val = y2[0:i]
            z2_val = z2[0:i]

            tq = t[0:i]

            v1_val = freq_1[0:i]
            v2_val = freq_2[0:i]
            
            if (i > when_to_track):
                line1.set_data(x1_val_track, y1_val_track)
                line1.set_3d_properties(z1_val_track)

                line2.set_data(x2_val_track, y2_val_track)
                line2.set_3d_properties(z2_val_track)

                point.set_3d_properties(z1[i])

            line1.set_color("red")
            line2.set_color("blue")

            line1_track.set_color("red")
            line2_track.set_color("blue")

            #------ x pro

            line3_1_x.set_data(tq, x1_val)
            point3_1_x.set_data(t[i], x1[i])
            line3_1_x.set_color("red")

            line3_2_x.set_data(tq, x2_val)
            point3_2_x.set_data(t[i], x2[i])
            line3_2_x.set_color("blue")

            #------ y pro

            line3_1_y.set_data(tq, y1_val)
            point3_1_y.set_data(t[i], y1[i])
            line3_1_y.set_color("red")

            line3_2_y.set_data(tq, y2_val)
            point3_2_y.set_data(t[i], y2[i])
            line3_2_y.set_color("blue")

            #------ z pro

            line3_1_z.set_data(tq, z1_val)
            point3_1_z.set_data(t[i], z1[i])
            line3_1_z.set_color("red")

            line3_2_z.set_data(tq, z2_val)
            point3_2_z.set_data(t[i], z2[i])
            line3_2_z.set_color("blue")

            #------ freq

            # line_v1.set_data(tq, v1_val)
            # point_v1.set_data(t[i], freq_1[i])
            # line_v1.set_color("red")

            # line_v2.set_data(tq, v2_val)
            # point_v2.set_data(t[i], freq_2[i])
            # line_v2.set_color("blue")

        # --- from the when_to_track point
            if (i > when_to_track):
                line1_track.set_data(x1_val_track, y1_val_track)
                line2_track.set_data(x2_val_track, y2_val_track)

                point_track1.set_data(x1[i], y1[i])
                point_track2.set_data(x2[i], y2[i])
        
        # --- from the start
            # line1_track.set_data(x1_val, y1_val)
            # line2_track.set_data(x2_val, y2_val)

            # point_track1.set_data(x1[i], y1[i])
            # point_track2.set_data(x2[i], y2[i])
            
        # return line1, line2, line1_track, line2_track, line3_1_x, line3_2_x, line3_1_y, line3_2_y, line3_1_z, line3_2_z, line_v1, line_v2, point, point_track1, point_track2, point3_1_x, point3_2_x, point3_1_y, point3_2_y, point3_1_z, point3_2_z, point_v1, point_v2
        return line1, line2, line1_track, line2_track, line3_1_x, line3_2_x, line3_1_y, line3_2_y, line3_1_z, line3_2_z, point, point_track1, point_track2, point3_1_x, point3_2_x, point3_1_y, point3_2_y, point3_1_z, point3_2_z


    def update_phi(j):
        j = (j - 1) * 100 - 1

        if (j < steps_t & j < steps_fr):
            tq = t[0:j]

            phi_val = phi1[0:j]

            line_phi.set_data(tq, phi_val)
            point_phi.set_data(t[j], phi1[j])
            line_phi.set_color("blue")
        
        return line_phi, point_phi
    
    anim = FuncAnimation(fig, update_all, frames=np.size(x1), interval=2000, blit=True)
    anim_phi = FuncAnimation(fig, update_phi, frames=np.size(x1), interval=2000, blit=True)

    plt.show()
    plt.savefig('rosslerAttractor.png')


def main():
    a = 0.22
    b = 0.1
    c = 8.5

    w1=1.02
    w2=0.98

    t_ini=0
    t_fin=200
    h=0.0001
    numsteps=int((t_fin-t_ini)/h)
    t=np.linspace(t_ini,t_fin,numsteps)

    d_ini=0
    d_fin=0.32
    d_h=0.00000016
    d_numsteps=int((d_fin-d_ini)/d_h)
    d=np.linspace(d_ini,d_fin,d_numsteps)

    x1=np.zeros(numsteps)
    y1=np.zeros(numsteps)
    z1=np.zeros(numsteps)

    x2=np.zeros(numsteps)
    y2=np.zeros(numsteps)
    z2=np.zeros(numsteps)

    freq_1=np.zeros(numsteps)
    freq_2=np.zeros(numsteps)

    phase_diff_1=np.zeros(numsteps)

    x1[0]=3
    y1[0]=12
    z1[0]=5

    x2[0]=-15
    y2[0]=-3
    z2[0]=-15

    for k in range(numsteps-1):
        [x1[k+1],y1[k+1],z1[k+1],x2[k+1],y2[k+1],z2[k+1]]=num_rossler(x1[k],y1[k],z1[k],x2[k],y2[k],z2[k],t[k+1]-t[k],a,b,c,w1,w2,d[k])

    for k in range(d_numsteps-1):
        [freq_1[k+1],freq_2[k+1],phase_diff_1[k+1]]=num_rossler_freq(x1[k],y1[k],z1[k],x2[k],y2[k],z2[k],t[k+1]-t[k],a,b,c,w1,w2,d[k])

    visualize(x1, y1, z1, x2, y2, z2, t, freq_1, freq_2, t_fin, numsteps, d_numsteps)

    
if __name__ == "__main__":
    main()
