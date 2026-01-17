import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pinn import PINN
from sampling import assemble_batches_physical_LHS
from losses import loss_total
from plotting import (
    plot_sampling_points_physical,
    plot_solution_slices_nearest,
    plot_losses_history
)

config = {
    #compute
    'device': 'cuda',
    'dtype': 'float32',
    'seed': 123,

    'x_min': 0.0,             #[m]
    'x_max': 1.0,             #[m]
    't_min': 0.0,             #[s]
    't_max': 13257.8450891164,#[s]
    'L': 1.0,                 #[m]
    'alpha': 1.0e-5,          #[m^2/s]
    'T_ref': 298.0,           #[K]
    'delta_T': 80.0,          #[K]
    'T0_phys': 333.0,         #K]
    'T_left_phys': 373.0,     #K]
    'T_right_phys': 298.0,    #[K]

    #sampling
    'n_f': 5000,
    'n_ic': 1000,
    'n_bc': 2000,

    #network
    'n_hidden': 32,
    'n_layers': 6,
    'activation': 'tanh',

    #training
    'epochs': 5000,
    'print_every': 10,
    'lr': 1e-4,
    
    #matlab reference comparison
    'mat_path': 'heat_data.mat',
    'mat_x_key': 'x',
    'mat_t_key': 't',
    'mat_u_key': 'T_all',
    't_max_phys': None,

    'plot_x_min': 0.0,
    'plot_x_max': 1.0,
    'plot_n_x': 200,
    'mat_temp_unit': 'C',
    'plot_temp_unit': 'C',

    'LBFGS_epochs': 50,
    'print_every_lbfgs': 3,
}

torch.manual_seed(config['seed'])
device = torch.device(config['device'])
dtype = getattr(torch, config['dtype'])

data = assemble_batches_physical_LHS(
    n_f=config['n_f'], n_ic=config['n_ic'], n_bc=config['n_bc'],
    x_min=config['x_min'], x_max=config['x_max'],
    t_min=config['t_min'], t_max=config['t_max'],
    L=config['L'], alpha=config['alpha'],
    T_ref=config['T_ref'], delta_T=config['delta_T'],
    T0_phys=config['T0_phys'],
    T_left_phys=config['T_left_phys'],
    T_right_phys=config['T_right_phys'],
    device=device, dtype=dtype, seed=config['seed'],
)

x_f_phys = data['x_f'] * float(config['L'])
t_f_phys = data['t_f'] * (float(config['L'])**2) / float(config['alpha'])
x_ic_phys = data['x_ic'] * float(config['L'])
t_b_phys  = data['t_b']  * (float(config['L'])**2) / float(config['alpha'])

plot_sampling_points_physical(
    x_f_phys=x_f_phys,
    t_f_phys=t_f_phys,
    x_ic_phys=x_ic_phys,
    t_b_phys=t_b_phys,
    L=float(config['L'])
)

xt_norm = torch.cat([
    torch.cat([data['x_f'],  data['t_f']], dim=1),
    torch.cat([data['x_ic'], torch.zeros_like(data['x_ic'])], dim=1),
    torch.cat([torch.zeros_like(data['t_b']), data['t_b']], dim=1),
    torch.cat([torch.ones_like (data['t_b']), data['t_b']], dim=1),
], dim=0)

model = PINN(
    n_input=2, n_output=1,
    n_hidden=config['n_hidden'],
    n_layers=config['n_layers'],
    X_for_norm=xt_norm,
    activation=config['activation'],
).to(device=device, dtype=dtype)

opt = torch.optim.Adam(model.parameters(), lr=config['lr'])

def to_kelvin(T, unit: str):
    unit_l = unit.lower()
    if unit_l in ['c', 'celsius', 'degc', '°c']:
        return T + 273.15
    if unit_l in ['k', 'kelvin']:
        return T
    return T

def load_mat_grid(cfg, device, dtype):
    m = loadmat(cfg['mat_path'])
    x_mat = torch.as_tensor(m[cfg['mat_x_key']].squeeze(), device=device, dtype=dtype)
    t_mat = torch.as_tensor(m[cfg['mat_t_key']].squeeze(), device=device, dtype=dtype)
    U = torch.as_tensor(m[cfg['mat_u_key']], device=device, dtype=dtype)
    T_K = to_kelvin(U, cfg['mat_temp_unit'])
    u_norm = (T_K - float(cfg['T_ref'])) / float(cfg['delta_T'])
    return x_mat, t_mat, u_norm

def nearest_indices_1d(grid, query):
    return torch.searchsorted(grid, query) - 1

x_mat_grid, t_mat_grid, u_mat_grid = load_mat_grid(config, device, dtype)

plot_solution_slices_nearest(
    model=model,
    mat_path=config['mat_path'],
    mat_x_key=config['mat_x_key'],
    mat_t_key=config['mat_t_key'],
    mat_u_key=config['mat_u_key'],
    L=float(config['L']),
    alpha=float(config['alpha']),
    T_ref=float(config['T_ref']),
    delta_T=float(config['delta_T']),
    t_max_phys=(None if config['t_max_phys'] is None else float(config['t_max_phys'])),
    x_min=float(config['plot_x_min']),
    x_max=float(config['plot_x_max']),
    n_x=int(config['plot_n_x']),
    device=data['x_f'].device,
    temp_unit=config['plot_temp_unit'],
    mat_temp_unit=config['mat_temp_unit']
)

hist = {'total': [], 'pde': [], 'ic': [], 'bc': []}



for epoch in range(config['epochs']+1):
    
    loss_val, parts = loss_total(model, data)
    
    opt.zero_grad(set_to_none=True)
    loss_val.backward()
    opt.step()

    hist['total'].append(float(loss_val.item()))
    hist['pde'].append(float(parts['pde'].item()))
    hist['ic'].append(float(parts['ic'].item()))
    hist['bc'].append(float(parts['bc'].item()))

    if (epoch + 1) % config['print_every'] == 0:
        print(f"Epoch {epoch+1:4d} | "
              f"Total={hist['total'][-1]:.4e} "
              f"pde={hist['pde'][-1]:.4e} "
              f"ic={hist['ic'][-1]:.4e} "
              f"bc={hist['bc'][-1]:.4e}")

plot_losses_history(hist)

plot_solution_slices_nearest(
    model=model,
    mat_path=config['mat_path'],
    mat_x_key=config['mat_x_key'],
    mat_t_key=config['mat_t_key'],
    mat_u_key=config['mat_u_key'],
    L=float(config['L']),
    alpha=float(config['alpha']),
    T_ref=float(config['T_ref']),
    delta_T=float(config['delta_T']),
    t_max_phys=(None if config['t_max_phys'] is None else float(config['t_max_phys'])),
    x_min=float(config['plot_x_min']),
    x_max=float(config['plot_x_max']),
    n_x=int(config['plot_n_x']),
    device=data['x_f'].device,
    temp_unit=config['plot_temp_unit'],
    mat_temp_unit=config['mat_temp_unit']
)

lbfgs_hist = {'total': [], 'pde': [], 'ic': [], 'bc': []}

lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0, max_iter=20, history_size=100,
    tolerance_grad=1e-7, tolerance_change=1e-9,
    line_search_fn='strong_wolfe'
)

for k in range(config['LBFGS_epochs']+1):
    
    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss_val, _ = loss_total(model, data)
        loss_val.backward()
        return loss_val

    lbfgs.step(closure)

    loss_val, parts = loss_total(model, data)
        
    lbfgs_hist['total'].append(float(loss_val.item()))
    lbfgs_hist['pde'].append(float(parts['pde'].item()))
    lbfgs_hist['ic'].append(float(parts['ic'].item()))
    lbfgs_hist['bc'].append(float(parts['bc'].item()))

    if (k + 1) % config['print_every_lbfgs'] == 0:
        print(f"[LBFGS] {k+1:03d} | "
            f"Total={lbfgs_hist['total'][-1]:.4e} "
            f"pde={lbfgs_hist['pde'][-1]:.4e} "
            f"ic={lbfgs_hist['ic'][-1]:.4e} "
            f"bc={lbfgs_hist['bc'][-1]:.4e}")

plot_losses_history(lbfgs_hist)

plot_solution_slices_nearest(
    model=model,
    mat_path=config['mat_path'],
    mat_x_key=config['mat_x_key'],
    mat_t_key=config['mat_t_key'],
    mat_u_key=config['mat_u_key'],
    L=float(config['L']),
    alpha=float(config['alpha']),
    T_ref=float(config['T_ref']),
    delta_T=float(config['delta_T']),
    t_max_phys=(None if config['t_max_phys'] is None else float(config['t_max_phys'])),
    x_min=float(config['plot_x_min']),
    x_max=float(config['plot_x_max']),
    n_x=int(config['plot_n_x']),
    device=data['x_f'].device,
    temp_unit=config['plot_temp_unit'],
    mat_temp_unit=config['mat_temp_unit']
)