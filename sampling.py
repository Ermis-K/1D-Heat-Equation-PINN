import numpy as np
import torch
from scipy.stats import qmc

def sample_interior_physical_lhs(n_f, x_min, x_max, t_min, t_max, device=None, dtype=torch.float32, seed=None):
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=2, seed=rng)
    U = torch.tensor(sampler.random(n_f), dtype=dtype, device=device)
    x = x_min + (x_max - x_min) * U[:, 0:1]
    t = t_min + (t_max - t_min) * U[:, 1:2]
    return x, t

def sample_ic_physical_lhs(n_ic, x_min, x_max, T0_phys, device=None, dtype=torch.float32, seed=None):
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=1, seed=rng)
    U = torch.tensor(sampler.random(n_ic), dtype=dtype, device=device)
    x = x_min + (x_max - x_min) * U[:, 0:1]
    T0 = torch.full_like(x, float(T0_phys))
    return x, T0

def sample_bc_times_physical_lhs(n_bc, t_min, t_max, device=None, dtype=torch.float32, seed=None):
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=1, seed=rng)
    U = torch.tensor(sampler.random(n_bc), dtype=dtype, device=device)
    t = t_min + (t_max - t_min) * U[:, 0:1]
    return t

def to_nd_inputs(x_phys, t_phys, L, alpha):
    x_nd = x_phys / L
    t_nd = (alpha * t_phys) / (L**2)
    return x_nd, t_nd

def to_nd_temperature(T_phys, T_ref, delta_T):
    return (T_phys - T_ref) / delta_T

def assemble_batches_physical_LHS(n_f, n_ic, n_bc,x_min, x_max, t_min, t_max,L, alpha,T_ref, delta_T,T0_phys, T_left_phys, T_right_phys,device=None, dtype=torch.float32, seed=None):

    x_f_phys, t_f_phys = sample_interior_physical_lhs(n_f, x_min, x_max, t_min, t_max, device, dtype, seed)
    x_ic_phys, T0_phys_tensor = sample_ic_physical_lhs(n_ic, x_min, x_max, T0_phys, device, dtype, seed)
    t_b_phys = sample_bc_times_physical_lhs(n_bc, t_min, t_max, device, dtype, seed)

    x_f,  t_f  = to_nd_inputs(x_f_phys, t_f_phys, L, alpha)
    x_ic, _    = to_nd_inputs(x_ic_phys, torch.zeros_like(x_ic_phys), L, alpha)
    _,    t_b  = to_nd_inputs(torch.zeros_like(t_b_phys), t_b_phys, L, alpha)

    u0      = to_nd_temperature(T0_phys_tensor,     T_ref, delta_T)
    g_left  = torch.full_like(t_b, float((T_left_phys  - T_ref) / delta_T))
    g_right = torch.full_like(t_b, float((T_right_phys - T_ref) / delta_T))

    return {'x_f': x_f, 't_f': t_f, 'x_ic': x_ic, 'u0': u0, 't_b': t_b, 'g_left': g_left, 'g_right': g_right}
