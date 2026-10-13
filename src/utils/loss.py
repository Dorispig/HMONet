"""Documentation string."""

__all__ = [
    "ComputeWinslowLoss",
]
# ============================================================
# Implementation comment.
# ============================================================


# ============================================================
# Implementation comment.
# ============================================================
import torch

# ============================================================
# Implementation comment.
# ============================================================


# ============================================================
# Implementation comment.
# ============================================================


# ============================================================
# Implementation comment.
# ============================================================
# Implementation comment.
# Implementation comment.
def ComputeWinslowLoss(x, y, xi, eta, device):
    # Implementation comment.
    grad_x_xi = torch.autograd.grad(
        x,
        xi,
        grad_outputs=torch.ones_like(x).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]
    grad_y_xi = torch.autograd.grad(
        y,
        xi,
        grad_outputs=torch.ones_like(y).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]

    grad_x_eta = torch.autograd.grad(
        x,
        eta,
        grad_outputs=torch.ones_like(x).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]
    grad_y_eta = torch.autograd.grad(
        y,
        eta,
        grad_outputs=torch.ones_like(y).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Implementation comment.
    grad_x_xi_xi = torch.autograd.grad(
        grad_x_xi,
        xi,
        grad_outputs=torch.ones_like(grad_x_xi).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]
    grad_y_xi_xi = torch.autograd.grad(
        grad_y_xi,
        xi,
        grad_outputs=torch.ones_like(grad_y_xi).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]

    grad_x_eta_eta = torch.autograd.grad(
        grad_x_eta,
        eta,
        grad_outputs=torch.ones_like(grad_x_eta).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]
    grad_y_eta_eta = torch.autograd.grad(
        grad_y_eta,
        eta,
        grad_outputs=torch.ones_like(grad_y_eta).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Implementation comment.
    grad_x_xi_eta = torch.autograd.grad(
        grad_x_xi,
        eta,
        grad_outputs=torch.ones_like(grad_x_xi).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]
    grad_y_xi_eta = torch.autograd.grad(
        grad_y_xi,
        eta,
        grad_outputs=torch.ones_like(grad_y_xi).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]

    # Implementation comment.
    alpha = grad_x_eta**2 + grad_y_eta**2
    beta = grad_x_xi * grad_x_eta + grad_y_xi * grad_y_eta
    gamma = grad_x_xi**2 + grad_y_xi**2

    # Implementation comment.
    pde_loss_x = (
        alpha * grad_x_xi_xi - 2 * beta * grad_x_xi_eta + gamma * grad_x_eta_eta
    )
    pde_loss_y = (
        alpha * grad_y_xi_xi - 2 * beta * grad_y_xi_eta + gamma * grad_y_eta_eta
    )

    return pde_loss_x, pde_loss_y

