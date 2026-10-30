"""Simplex and MM numerical primitives used by the joint models."""
import torch

FLOOR = 1e-8


def simplex(v):
    z = v-FLOOR
    u = torch.sort(z, dim=1, descending=True).values
    css = torch.cumsum(u, dim=1)-(1-v.shape[1]*FLOOR)
    ind = torch.arange(1, v.shape[1]+1, device=v.device, dtype=v.dtype)
    rho = (u-css/ind > 0).sum(1)-1
    theta = css[torch.arange(len(v), device=v.device), rho]/(rho+1)
    return (z-theta[:, None]).clamp_min(0)+FLOOR


def em_simplex(a):
    """Exact historical M-step on the floor-constrained unit simplex."""
    u = torch.sort(a, dim=1, descending=True).values
    index = torch.arange(1, a.shape[1]+1, device=a.device, dtype=a.dtype)
    candidate = torch.cumsum(u, dim=1) / (1-(a.shape[1]-index)*FLOOR)
    active = (u > FLOOR*candidate).sum(1)-1
    multiplier = candidate[torch.arange(len(a), device=a.device), active]
    return (a / multiplier[:, None]).clamp_min(FLOOR)


def simplex_mass(v, mass):
    """Euclidean projection with every coordinate >= FLOOR and row sum mass."""
    z = v - FLOOR
    u = torch.sort(z, dim=1, descending=True).values
    css = torch.cumsum(u, dim=1) - (mass - v.shape[1] * FLOOR)
    index = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype)
    rho = (u - css / index > 0).sum(1) - 1
    theta = css[torch.arange(len(v), device=v.device), rho] / (rho + 1)
    return (z - theta[:, None]).clamp_min(0) + FLOOR


def em_mass(a, mass):
    """Exact M-step: maximize sum(a log c), sum(c)=mass, c>=FLOOR."""
    u = torch.sort(a, dim=1, descending=True).values
    index = torch.arange(1, a.shape[1]+1, device=a.device, dtype=a.dtype)
    candidate = torch.cumsum(u, dim=1) / (mass-(a.shape[1]-index)*FLOOR)
    active = (u > FLOOR*candidate).sum(1)-1
    multiplier = candidate[torch.arange(len(a), device=a.device), active]
    return (a / multiplier[:, None]).clamp_min(FLOOR)


def quadratic_log_simplex(a, curvature, linear):
    """Minimize -a.log(p)+c/2*||p||^2+linear.p under the same P simplex."""
    m = a.shape[1]
    c = curvature[:, None]
    reference_multiplier = a*m-c/m-linear
    lo = reference_multiplier.min(1).values[:, None]
    hi = reference_multiplier.max(1).values[:, None]
    def roots(multiplier):
        b = linear+multiplier
        disc = torch.sqrt(b.square()+4*c*a)
        value = torch.where(b >= 0, 2*a/(b+disc).clamp_min(1e-300),
                            (disc-b)/(2*c).clamp_min(1e-300))
        return value.clamp_min(FLOOR)
    for _ in range(45):
        mid = (lo+hi)/2
        value = roots(mid)
        too_much = value.sum(1, keepdim=True) > 1
        lo = torch.where(too_much, mid, lo)
        hi = torch.where(too_much, hi, mid)
    p = roots((lo+hi)/2)
    zero = curvature <= 1e-30
    if bool(zero.any()):
        p[zero] = em_simplex(a[zero])
    return p
