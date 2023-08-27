import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import hilbert

import math

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

        
def visualize(x, y, z, t, phi, v):
    fig = plt.figure(figsize=(10, 6))  

    h1 = hilbert(x)
    phi1 = np.unwrap(np.angle(h1))
    
    axis_1 = fig.add_subplot(1, 2, 2, projection='3d')
    line, = axis_1.plot([], [], [], color='b', linewidth=0.8)
    
    point, = axis_1.plot([], [], [], marker='o', color='b', markersize=4)
    point2, = axis_1.plot([], [], [], marker='o', color='r', markersize=4)
    
    axis_1.xaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    axis_1.yaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    axis_1.zaxis.set_pane_color((1.0, 0.0, 1.0, 0.0))
    
    axis_1.set_xlim(-28, 28)
    axis_1.set_ylim(-28, 28)
    axis_1.set_zlim(0, 50)
    
    axis_1.set_xlabel('X', fontsize=12)
    axis_1.set_ylabel('Y', fontsize=12)
    axis_1.set_zlabel('Z', fontsize=12)
    
    axis_1.set_yticklabels([])
    axis_1.set_xticklabels([])
    axis_1.set_zticklabels([])

    axis_2 = fig.add_subplot(3, 2, 1)
    line3, = axis_2.plot([], [], 'b-', linewidth=0.8)
 
    point3, = axis_2.plot([], [], marker='o', color='b', markersize=4)

    axis_2.set_xlim(t[0], t[len(t) - 1])
    axis_2.set_ylim(-20, 30)
    axis_2.set_xlim(0, 200)
    axis_2.set_ylabel('x(t)', fontsize=8)

    axis_3 = fig.add_subplot(3, 2, 5)
    line_phi, = axis_3.plot([], [], 'g-', linewidth=0.8)

    axis_3.set_xlim(0, 200)
    axis_3.set_ylim(-50, 250)
    axis_3.set_ylabel('phi(t) - фаза', fontsize=8)
    point_phi, = axis_3.plot([], [], marker='o', color='r', markersize=4)

    axis_4 = fig.add_subplot(3, 2, 3)
    axis_4.set_ylabel('v(t) - мнгновенная частота', fontsize=8)
    axis_4.set_ylim(-20, 30)
    axis_4.set_xlim(0, 200)
    line_v, = axis_4.plot([], [], 'g-', linewidth=0.8)
    phi_max = max(phi)
    v_max = max(v)

    # axis_4.set_xlim(0, 90)
    # axis_4.set_ylim(0, 1.5)
    point_v, = axis_4.plot([], [], marker='o', color='r', markersize=4)
    
    def update_all(i):
        i = i * 600

        x_val = x[0:i]
        y_val = y[0:i]
        z_val = z[0:i]

        tq = t[0:i]

        phi_val = phi[0:i]
        v_val = v[0:i]
        
        line.set_data(x_val, y_val)
        line.set_3d_properties(z_val)
        line.set_3d_properties(z_val)

        point.set_3d_properties(z[i])
        line.set_color("darkcyan")

        line3.set_data(tq, x_val)
        point3.set_data(t[i], x[i])
        line3.set_color("red")

        line_v.set_data(tq, v_val)
        point_v.set_data(t[i], v[i])
        line_v.set_color("green")
        
        return line, line3, line_v, point, point3, point_v
    
    def update_phi(j):
        j = j * 600

        tq = t[0:j]

        phi_val = phi1[0:j]

        line_phi.set_data(tq, phi_val)
        point_phi.set_data(t[j], phi1[j])
        line_phi.set_color("blue")
        
        return line_phi, point_phi
    
    anim = FuncAnimation(fig, update_all, frames=np.size(x), interval=0, blit=True)
    anim_phi = FuncAnimation(fig, update_phi, frames=np.size(x), interval=0, blit=True)

    plt.savefig('rosslerAttractor.png')
    plt.show()


def main():
    t = np.arange(0, 300, 0.001)
    
    x_0 = np.zeros((len(t)))
    y_0 = np.zeros((len(t)))
    z_0 = np.zeros((len(t)))

    phi = np.zeros((len(t)))
    v = np.zeros((len(t)))

    a = 0.22
    b = 0.1
    c = 8.5
    w = 0.98
    
    x_0[0] = 0
    y_0[0] = 0
    z_0[0] = 0
    phi[0] = 0
    v[0] = 0
    
    solve(x_0, y_0, z_0, phi, v, t, a, b, c, w)
    visualize(x_0, y_0, z_0, t, phi, v)
  
    
if __name__ == "__main__":
    main()
