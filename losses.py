import torch

def pde_residual_nd(model, x_nd, t_nd):
    xt = torch.cat((x_nd, t_nd), dim=-1).requires_grad_(True)
    u = model(xt)
    grad = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_X   = grad[..., 0:1]
    u_tau = grad[..., 1:2]
    u_XX  = torch.autograd.grad(u_X, xt, grad_outputs=torch.ones_like(u_X), create_graph=True)[0][..., 0:1]
    
    return u_tau - u_XX

def loss_pde(model, x_f, t_f):
    r = pde_residual_nd(model, x_f, t_f)

    return torch.mean(r**2)

def loss_ic(model, x_ic, theta0_target):
    t0 = torch.zeros_like(x_ic)
    u0 = model(torch.cat((x_ic, t0), dim=-1))

    return torch.mean((u0 - theta0_target)**2)

def loss_bc_dirichlet_both_ends(model, t_b, g_left, g_right):
    x0 = torch.zeros_like(t_b)
    x1 = torch.ones_like(t_b)
    uL = model(torch.cat((x0, t_b), dim=-1))
    uR = model(torch.cat((x1, t_b), dim=-1))

    return 0.5*(torch.mean((uL - g_left)**2) + torch.mean((uR - g_right)**2))

def loss_total(model, batches,lambda_bc, lambda_ic):
    L_pde = loss_pde(model, batches['x_f'],  batches['t_f'])
    L_ic  = loss_ic (model, batches['x_ic'], batches['u0'])
    L_bc  = loss_bc_dirichlet_both_ends(model, batches['t_b'], batches['g_left'], batches['g_right'])
    L_tot = L_pde + lambda_ic*L_ic + lambda_bc*L_bc

    return L_tot, {'pde': L_pde, 'ic': L_ic, 'bc': L_bc}