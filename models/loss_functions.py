import torch


def dist_loss(points):
    P = points
    Pb = P.roll(1, dims=2)

    D = (P - Pb) ** 2

    return torch.sum(D, dim=[-2, -1]).mean()


def curvature_loss(points):
    P = points
    Pf = P.roll(-1, dims=2)
    Pb = P.roll(1, dims=2)

    K = Pf + Pb - 2 * P

    return torch.norm(K, dim=-1).mean()


class Grad(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult
        super().__init__()

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad
