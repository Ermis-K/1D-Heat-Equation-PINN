import torch
from pinn import PINN
from sampling import assemble_batches_physical_LHS
from losses import loss_total
from plotting import (
    plot_sampling_points_physical,
    plot_solution_slices_nearest,
    plot_losses_history)
from scipy.io import loadmat
from adaptive_weights import adaptive_weights


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
    'n_f': 20000,
    'n_ic': 1000,
    'n_bc': 2000,

    #network
    'n_hidden': 128,
    'n_layers': 6,
    'activation': 'tanh',

    #training
    'epochs': 1000,
    'print_every': 10,
    'lr': 1e-4,
    'batch_frac_f': 1,
    'batch_frac_ic': 1,
    'batch_frac_bc': 1,

    #matlab reference comparison
    'mat_path': 'heat_data.mat',
    'mat_x_key': 'x',
    'mat_t_key': 't',
    'mat_u_key': 'T_all',
    #leave t_max_phys none to infer from mat
    't_max_phys': None,

    'plot_x_min': 0.0,
    'plot_x_max': 1.0,
    'plot_n_x': 200,
    'mat_temp_unit': 'C',
    'plot_temp_unit': 'C',

    'warmup_epochs': 0,

    'LBFGS_epochs': 20,
    'print_every_lbfgs': 2,

    #initial lambda weights
    'lambda_ic': 1,
    'lambda_bc': 1,
    'step_interval': 10,
    'step_interval_lbfgs': 5,
    'alpha_weights': 0.5, #me 0, praktika apenergopoiountai ta adaptive weights, 0.9 default
}

torch.manual_seed(config['seed'])
device = torch.device(config['device'])
dtype = getattr(torch, config['dtype'])

batches = assemble_batches_physical_LHS(
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

x_f_phys = batches['x_f'] * float(config['L'])
t_f_phys = batches['t_f'] * (float(config['L'])**2) / float(config['alpha'])

x_ic_phys = batches['x_ic'] * float(config['L'])
t_b_phys  = batches['t_b']  * (float(config['L'])**2) / float(config['alpha'])

plot_sampling_points_physical(
    x_f_phys=x_f_phys,
    t_f_phys=t_f_phys,
    x_ic_phys=x_ic_phys,
    t_b_phys=t_b_phys,
    L=float(config['L'])
)

xt_norm = torch.cat([
    torch.cat([batches['x_f'],  batches['t_f']], dim=1),
    torch.cat([batches['x_ic'], torch.zeros_like(batches['x_ic'])], dim=1),
    torch.cat([torch.zeros_like(batches['t_b']), batches['t_b']], dim=1),
    torch.cat([torch.ones_like (batches['t_b']), batches['t_b']], dim=1),
], dim=0)

model = PINN(
    n_input=2, n_output=1,
    n_hidden=config['n_hidden'],
    n_layers=config['n_layers'],
    X_for_norm=xt_norm,
    activation=config['activation'],
).to(device=device, dtype=dtype)

opt = torch.optim.Adam(model.parameters(), lr=config['lr'])

Nf  = batches['x_f'].shape[0]
Nic = batches['x_ic'].shape[0]
Nbc = batches['t_b'].shape[0]

Bf  = max(1, int(config['batch_frac_f']  * Nf))
Bic = max(1, int(config['batch_frac_ic'] * Nic))
Bbc = max(1, int(config['batch_frac_bc'] * Nbc))

def make_chunks(n, B, device_):
    perm = torch.randperm(n, device=device_)
    return [perm[i:i+B] for i in range(0, n, B)]

#mat grid loader, nearest lookup for warmup
def _to_kelvin(T, unit: str):
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
    T_K = _to_kelvin(U, cfg['mat_temp_unit'])
    u_norm = (T_K - float(cfg['T_ref'])) / float(cfg['delta_T'])  # dimensionless
    return x_mat, t_mat, u_norm

def nearest_indices_1d(grid, query):
    return torch.searchsorted(grid, query) - 1

x_mat_grid, t_mat_grid, u_mat_grid = load_mat_grid(config, device, dtype)

warmup_hist = []

for epoch in range(int(config['warmup_epochs'])):
    chunks_f = make_chunks(Nf, Bf, device)
    n_steps  = len(chunks_f)

    epoch_loss_sum = 0.0

    for step in range(n_steps):
        idx_f = chunks_f[step % len(chunks_f)]
        x_n = batches['x_f'][idx_f]  #normalized in [0,1]
        t_n = batches['t_f'][idx_f]  #normalized in [0,1]

        x_phys = x_n * float(config['L'])
        t_phys = t_n * (float(config['L'])**2) / float(config['alpha'])

        ix = nearest_indices_1d(x_mat_grid, x_phys)
        it = nearest_indices_1d(t_mat_grid, t_phys)

        u_target = u_mat_grid[it, ix].reshape(-1, 1)  #[Bf,1]

        xt = torch.cat([x_n, t_n], dim=1)  #[Bf,2]
        pred = model(xt)

        loss_sup = torch.mean((pred - u_target) ** 2)

        opt.zero_grad(set_to_none=True)
        loss_sup.backward()
        opt.step()

        epoch_loss_sum += float(loss_sup.item())

    warmup_hist.append(epoch_loss_sum / n_steps)

    if (epoch + 1) % config['print_every'] == 0:
        print(f"[WARM-UP] Epoch {epoch+1:4d} | supervised_mse={warmup_hist[-1]:.4e}")

#============================================================================

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
    device=batches['x_f'].device,
    temp_unit=config['plot_temp_unit'],
    mat_temp_unit=config['mat_temp_unit']
)

hist = {'total': [], 'pde': [], 'ic': [], 'bc': []}

weights_hist={'ic': [], 'bc': []}

lambda_ic = torch.tensor(float(config['lambda_ic']), device=device, dtype=dtype)
lambda_bc = torch.tensor(float(config['lambda_bc']), device=device, dtype=dtype)


for epoch in range(config['epochs']):
    chunks_f  = make_chunks(Nf,  Bf,  device)
    chunks_ic = make_chunks(Nic, Bic, device)
    chunks_bc = make_chunks(Nbc, Bbc, device)
    n_steps = max(len(chunks_f), len(chunks_ic), len(chunks_bc))

    tot_loss_sum = 0.0
    pde_loss_sum = 0.0
    ic_loss_sum  = 0.0
    bc_loss_sum  = 0.0

    for step in range(n_steps):
        idx_f  = chunks_f [step % len(chunks_f )]
        idx_ic = chunks_ic[step % len(chunks_ic)]
        idx_bc = chunks_bc[step % len(chunks_bc)]

        batch = {
            'x_f':   batches['x_f'][idx_f],
            't_f':   batches['t_f'][idx_f],
            'x_ic':  batches['x_ic'][idx_ic],
            'u0':    batches['u0'][idx_ic],
            't_b':   batches['t_b'][idx_bc],
            'g_left':  batches['g_left'][idx_bc],
            'g_right': batches['g_right'][idx_bc],
        }

        _ , parts = loss_total(model, batch, lambda_bc, lambda_ic)
        loss_pde=parts['pde']
        loss_ic=parts['ic']
        loss_bc=parts['bc']

        if epoch % config['step_interval'] == 0:
            lambda_ic, lambda_bc = adaptive_weights(
                model, loss_pde, loss_bc, loss_ic, lambda_ic, lambda_bc, config['alpha_weights']
            )

            weights_hist['ic'].append(lambda_ic.item())
            weights_hist['bc'].append(lambda_bc.item())

        L_tot=loss_pde + lambda_bc*loss_bc + lambda_ic*loss_ic

        opt.zero_grad(set_to_none=True)
        L_tot.backward()
        opt.step()


        tot_loss_sum += float(L_tot.item())
        pde_loss_sum += float(parts['pde'].item())
        ic_loss_sum  += float(parts['ic'].item())
        bc_loss_sum  += float(parts['bc'].item())

    hist['total'].append(tot_loss_sum / n_steps)
    hist['pde'].append(pde_loss_sum / n_steps)
    hist['ic'].append(ic_loss_sum  / n_steps)
    hist['bc'].append(bc_loss_sum  / n_steps)

    if (epoch + 1) % config['print_every'] == 0:
        print(f"Epoch {epoch+1:4d} | "
              f"total={hist['total'][-1]:.4e} "
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
    device=batches['x_f'].device,
    temp_unit=config['plot_temp_unit'],
    mat_temp_unit=config['mat_temp_unit']
)

full_batch = {
    'x_f': batches['x_f'], 't_f': batches['t_f'],
    'x_ic': batches['x_ic'], 'u0': batches['u0'],
    't_b': batches['t_b'], 'g_left': batches['g_left'], 'g_right': batches['g_right'],
}

lbfgs_hist = {'total': [], 'pde': [], 'ic': [], 'bc': []}

weights_hist_lbfgs = {'ic': [], 'bc': []}

lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0, max_iter=100, history_size=100,
    tolerance_grad=1e-7, tolerance_change=1e-9,
    line_search_fn='strong_wolfe'
)

lambda_ic = torch.tensor(weights_hist['ic'][-1], device=device, dtype=dtype)
lambda_bc = torch.tensor(weights_hist['bc'][-1], device=device, dtype=dtype)

for k in range(config['LBFGS_epochs']):
    did_update = [False]

    def closure():
        lbfgs.zero_grad(set_to_none=True)

        #component losses without total
        _, parts = loss_total(model, full_batch, lambda_bc, lambda_ic)
        loss_pde, loss_ic_val, loss_bc_val = parts['pde'], parts['ic'], parts['bc']

        #update adaptive weights
        if (k % config['step_interval_lbfgs'] == 0) and (not did_update[0]):
            lic, lbc = adaptive_weights(
                model, loss_pde, loss_bc_val, loss_ic_val,
                lambda_ic, lambda_bc,
                alpha=config['alpha_weights']
            )
            lambda_ic.copy_(lic)
            lambda_bc.copy_(lbc)

            weights_hist_lbfgs['ic'].append(lambda_ic.item())
            weights_hist_lbfgs['bc'].append(lambda_bc.item())
            did_update[0] = True

        #total loss with current weights
        L_tot = loss_pde + lambda_bc * loss_bc_val + lambda_ic * loss_ic_val
        L_tot.backward()

        lbfgs_hist['total'].append(float(L_tot.detach().item()))
        lbfgs_hist['pde'].append(float(loss_pde.detach().item()))
        lbfgs_hist['ic'].append(float(loss_ic_val.detach().item()))
        lbfgs_hist['bc'].append(float(loss_bc_val.detach().item()))
        return L_tot

    lbfgs.step(closure)

    if (k + 1) % config['print_every_lbfgs'] == 0:
        print(f"[LBFGS] {k+1:03d} | "
            f"total={lbfgs_hist['total'][-1]:.4e} "
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
    device=batches['x_f'].device,
    temp_unit=config['plot_temp_unit'],
    mat_temp_unit=config['mat_temp_unit']
)

import matplotlib.pyplot as plt

#Adam weights---
plt.figure(figsize=(6,4))
plt.plot(weights_hist['ic'], label='Adam λ_ic')
plt.plot(weights_hist['bc'], label='Adam λ_bc')
plt.xlabel("Epochs")
plt.ylabel("λ value")
plt.yscale("log")
plt.legend()
plt.title("Adaptive weights during Adam")
plt.tight_layout()
plt.show()

#LBFGS weights---
plt.figure(figsize=(6,4))
plt.plot(weights_hist_lbfgs['ic'], label='LBFGS λ_ic')
plt.plot(weights_hist_lbfgs['bc'], label='LBFGS λ_bc')
plt.xlabel("Iterations")
plt.ylabel("λ value")
plt.yscale("log")
plt.legend()
plt.title("Adaptive weights during LBFGS")
plt.tight_layout()
plt.show()

import numpy as np

x_mat_grid, t_mat_grid, u_mat_grid = load_mat_grid(config, device, dtype)

X, T = torch.meshgrid(x_mat_grid, t_mat_grid, indexing='ij')
XT = torch.stack([X.flatten(), T.flatten()], dim=1).to(device=device, dtype=dtype)

#predictions
with torch.no_grad():
    u_pred_norm = model(XT).cpu().numpy().flatten()

#reference
u_true_norm = u_mat_grid.cpu().numpy().flatten()

mae = np.mean(np.abs(u_pred_norm - u_true_norm)) * float(config['delta_T'])

rel_l2 = np.linalg.norm(u_pred_norm - u_true_norm) / np.linalg.norm(u_true_norm) * 100

print("====================================================")
print(f"Mean Absolute Error = {mae:.3f} {config['plot_temp_unit']}")
print(f"Relative L2 Error  = {rel_l2:.2f} %")
print("====================================================")
