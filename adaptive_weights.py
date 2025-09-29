import torch

def _flatten_gradients(model):
    flat = []
    for p in model.parameters():
        if p.grad is not None:
            flat.append(p.grad.view(-1))
    if flat:
        return torch.cat(flat)
    else:
        return torch.tensor([0.0], device=next(model.parameters()).device)

def _compute_grads(model, loss):
    model.zero_grad()
    loss.backward(retain_graph=True)
    return _flatten_gradients(model).abs()

def adaptive_weights(model, loss_pde, loss_bc, loss_ic, lambda_ic, lambda_bc, alpha=0.9, eps=1e-12):
    g_r  = _compute_grads(model, loss_pde)
    g_bc = _compute_grads(model, loss_bc)
    g_ic = _compute_grads(model, loss_ic)

    g_r_max = g_r.max().item()
    g_bc_mean = g_bc.mean().item() + eps
    g_ic_mean = g_ic.mean().item() + eps

    hat_bc = g_r_max / g_bc_mean
    hat_ic = g_r_max / g_ic_mean

    # EMA update
    lambda_bc = (1 - alpha) * lambda_bc + alpha * hat_bc
    lambda_ic = (1 - alpha) * lambda_ic + alpha * hat_ic

    return lambda_ic.detach().to(next(model.parameters()).device), \
        lambda_bc.detach().to(next(model.parameters()).device)

