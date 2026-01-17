import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

def convert_T_unit(T_array, to_unit):
    if to_unit.upper() == 'K':
        return T_array
    elif to_unit.upper() == 'C':
        return T_array - 273.15

def predict_T_physical(model, x_phys, t_phys, L, alpha, T_ref, delta_T, device=None, dtype=torch.float32, temp_unit='K'):
    x_nd = torch.as_tensor(x_phys, dtype=dtype, device=device).reshape(-1, 1) / float(L)
    t_nd = (float(alpha) * torch.as_tensor(t_phys, dtype=dtype, device=device).reshape(-1, 1)) / (float(L) ** 2)
    xt = torch.cat((x_nd, t_nd), dim=-1)
    with torch.no_grad():
        u = model(xt)
    T_K = u.detach().cpu().numpy().squeeze() * float(delta_T) + float(T_ref)
    return convert_T_unit(T_K, temp_unit)

def plot_sampling_points_physical(x_f_phys, t_f_phys, x_ic_phys=None, t_b_phys=None, L=None):
    plt.figure(figsize=(6,5))
    plt.scatter(x_f_phys.cpu().numpy(), t_f_phys.cpu().numpy(), s=6, label='Interior (PDE)')
    if x_ic_phys is not None:
        zi = np.zeros_like(x_ic_phys.cpu().numpy())
        plt.scatter(x_ic_phys.cpu().numpy(), zi, s=10, label='IC (t=0)')
    if t_b_phys is not None:
        tb = t_b_phys.cpu().numpy()
        x0 = np.zeros_like(tb)
        x1 = np.full_like(tb, float(L))
        plt.scatter(x0, tb, s=10, label='BC: x=0')
        plt.scatter(x1, tb, s=10, label='BC: x=L')
    plt.xlabel('x [m]')
    plt.ylabel('t [s]')
    plt.title('Physical sampled points')
    plt.legend()
    plt.tight_layout()
    plt.show()

def load_mat_user_keys(mat_path):
    data = loadmat(mat_path)
    keys = [k for k in data.keys() if not k.startswith('__')]
    return data, keys

def plot_solution_slices_nearest(model, mat_path, mat_x_key, mat_t_key, mat_u_key, L, alpha, T_ref, delta_T, t_max_phys=None, x_min=0.0, x_max=None, n_x=200, device=None, dtype=torch.float32, temp_unit='K', mat_temp_unit='K'):
    data = loadmat(mat_path)
    x_mat = np.asarray(data[mat_x_key]).squeeze().astype(float)
    t_mat = np.asarray(data[mat_t_key]).squeeze().astype(float)
    U_mat = np.asarray(data[mat_u_key])

    nx, nt = x_mat.size, t_mat.size
    if U_mat.shape == (nt, nx):
        U_mat = U_mat.T

    if x_max is None:
        x_max = float(np.max(x_mat))
    if t_max_phys is None:
        t_max_phys = float(np.max(t_mat))

    x_eval = np.linspace(float(x_min), float(x_max), int(n_x), dtype=float)

    times = [0.0, 0.05 * t_max_phys, 0.1 * t_max_phys, 0.2 * t_max_phys]

    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    axs = axs.ravel()

    y_label = 'T [K]' if temp_unit.upper() == 'K' else 'T [°C]'

    for i, t_phys in enumerate(times):
        T_pred = predict_T_physical(model, x_eval, np.full_like(x_eval, t_phys, dtype=float), L=L, alpha=alpha, T_ref=T_ref, delta_T=delta_T, device=device, dtype=dtype, temp_unit=temp_unit)

        idx_t = int(np.argmin(np.abs(t_mat - t_phys)))
        U_t = U_mat[:, idx_t]

        T_true = np.interp(x_eval, x_mat, U_t)

        if mat_temp_unit.upper() == 'K':
            T_true = convert_T_unit(T_true, temp_unit)
        elif mat_temp_unit.upper() == 'C':
            if temp_unit.upper() == 'K':
                T_true = T_true + 273.15

        ax = axs[i]
        ax.plot(x_eval, T_true, linewidth=2, label='MAT solution')
        ax.plot(x_eval, T_pred, linestyle='--', linewidth=2, label='PINN')
        ax.set_title(f't = {t_mat[idx_t]:.3g} s (closest to {t_phys:.3g} s)')
        ax.set_xlabel('x [m]')
        ax.set_ylabel(y_label)
        ax.grid(True)
        if i == 0:
            ax.legend()

    fig.suptitle(f'Physical Temperature Slices: PINN vs .mat ({temp_unit.upper()})', y=0.98)
    plt.tight_layout()
    plt.show()

def plot_losses_history(history):
    plt.figure(figsize=(7,5))
    if 'total' in history: plt.plot(history['total'], label='total')
    if 'pde'   in history: plt.plot(history['pde'],   label='pde')
    if 'ic'    in history: plt.plot(history['ic'],    label='ic')
    if 'bc'    in history: plt.plot(history['bc'],    label='bc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss History')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()