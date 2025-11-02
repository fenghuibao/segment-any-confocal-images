# Author: Huibao Feng
# Date: 2025-11-02

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn.functional as F
import numpy as np

# ---------------------
# Public datatypes
# ---------------------

@dataclass
class SegmentationInfo:
    """Holds optimization traces for UI/preview."""
    iterations_run: int
    deltas: List[float]
    converged: bool


# ---------------------
# Math helpers
# ---------------------

_TINY = 1e-15
_DEFAULT_TOL = 1e-8


def _gaussian(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Element-wise Gaussian pdf."""
    return 1.0 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(-(x - mu) ** 2 / (2 * sigma**2))


def _calculate_resp(X: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """E-step responsibilities for a 1D GMM."""
    resp = pi * _gaussian(X.reshape(-1, 1), mu, sigma)
    return resp / (resp.sum(dim=1, keepdim=True) + _TINY)


def _loglh_gmm(X: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Log-likelihood of a 1D GMM."""
    return torch.log((pi * _gaussian(X.reshape(-1, 1), mu, sigma)).sum(dim=1) + _TINY).sum()


# ---------------------
# Pairwise & potentials
# ---------------------

def _get_adjacent_labels(image_label: torch.Tensor):
    """4-neighborhood for 2D (C,H,W) and 6-neighborhood for 3D (C,D,H,W)."""
    if image_label.ndim == 3:
        # (C,H,W)
        left = F.pad(image_label, (1, 0, 0, 0))[:, :, :-1]
        right = F.pad(image_label, (0, 1, 0, 0))[:, :, 1:]
        front = F.pad(image_label, (0, 0, 1, 0))[:, :-1, :]
        back = F.pad(image_label, (0, 0, 0, 1))[:, 1:, :]
        return left, right, front, back
    else:
        # (C,D,H,W)
        left = F.pad(image_label, (1, 0, 0, 0, 0, 0))[:, :, :, :-1]
        right = F.pad(image_label, (0, 1, 0, 0, 0, 0))[:, :, :, 1:]
        front = F.pad(image_label, (0, 0, 1, 0, 0, 0))[:, :, :-1, :]
        back = F.pad(image_label, (0, 0, 0, 1, 0, 0))[:, :, 1:, :]
        up = F.pad(image_label, (0, 0, 0, 0, 1, 0))[:, :-1, :, :]
        down = F.pad(image_label, (0, 0, 0, 0, 0, 1))[:, 1:, :, :]
        return left, right, front, back, up, down


def _smoothness_potential(label: torch.Tensor, smoothness_energy: float):
    # Potentials for two classes (0 / 1) given neighbor labels
    indicator_0 = torch.where(label == 0, -1, 1)
    indicator_1 = torch.where(label == 1, -1, 1)
    return indicator_0 * smoothness_energy, indicator_1 * smoothness_energy


def _frangi_potential(frangi: torch.Tensor, beta1: float):
    f0 = frangi / (frangi.max() + _TINY)
    f1 = 1.0 - f0
    # add tiny for stability
    return beta1 * torch.log(f0 + _TINY), beta1 * torch.log(f1 + _TINY)


def _pairwise_potential(
    image_label: torch.Tensor,
    beta2: float,
    sigma: float,
    frangi_pot0: torch.Tensor,
    frangi_pot1: torch.Tensor,
    device: torch.device,
    pixel_size_xy: float,
    pixel_size_z: Optional[float] = None,
) -> torch.Tensor:
    """Return unary+pairwise potentials for both classes stacked along first dim."""
    smooth_xy = float(beta2 * np.exp(-pixel_size_xy ** 2 / (2.0 * sigma ** 2)))
    if image_label.ndim == 3:
        left, right, front, back = _get_adjacent_labels(image_label)
        p0_l, p1_l = _smoothness_potential(left, smooth_xy)
        p0_r, p1_r = _smoothness_potential(right, smooth_xy)
        p0_f, p1_f = _smoothness_potential(front, smooth_xy)
        p0_b, p1_b = _smoothness_potential(back, smooth_xy)
        pot0 = p0_l + p0_r + p0_f + p0_b + frangi_pot0
        pot1 = p1_l + p1_r + p1_f + p1_b + frangi_pot1
    else:
        assert pixel_size_z is not None, "pixel_size_z required for 3D volumes"
        smooth_z = float(beta2 * np.exp(-pixel_size_z ** 2 / (2.0 * sigma ** 2)))
        left, right, front, back, up, down = _get_adjacent_labels(image_label)
        p0_l, p1_l = _smoothness_potential(left, smooth_xy)
        p0_r, p1_r = _smoothness_potential(right, smooth_xy)
        p0_f, p1_f = _smoothness_potential(front, smooth_xy)
        p0_b, p1_b = _smoothness_potential(back, smooth_xy)
        p0_u, p1_u = _smoothness_potential(up, smooth_z)
        p0_d, p1_d = _smoothness_potential(down, smooth_z)
        pot0 = p0_l + p0_r + p0_f + p0_b + p0_u + p0_d + frangi_pot0
        pot1 = p1_l + p1_r + p1_f + p1_b + p1_u + p1_d + frangi_pot1

    return torch.concat([pot0, pot1]).to(device)


# ---------------------
# GMM parameter flows
# ---------------------

def _parameter_initialization(
    device: torch.device,
    pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
    data_fore: torch.Tensor, data_back: torch.Tensor,
    n_fore: int, n_back: int,
):
    """KMeans init mixed with wide sigma, uniform pi."""
    # Move to CPU for sklearn, then back:
    from sklearn.cluster import KMeans  # lazy import to keep the top clean

    mu[:n_back, 0] = torch.tensor(
        KMeans(n_clusters=n_back, n_init="auto").fit(data_back.detach().cpu().reshape(-1, 1)).cluster_centers_.reshape(-1),
        dtype=torch.float32, device=device
    )
    mu[:n_fore, 1] = torch.tensor(
        KMeans(n_clusters=n_fore, n_init="auto").fit(data_fore.detach().cpu().reshape(-1, 1)).cluster_centers_.reshape(-1),
        dtype=torch.float32, device=device
    )
    sigma[:n_back, 0] = 256
    sigma[:n_fore, 1] = 256
    pi[:n_back, 0] = 1.0 / n_back
    pi[:n_fore, 1] = 1.0 / n_fore
    return pi, mu, sigma


def _switch_parameters(
    data: torch.Tensor,
    pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
    label: torch.Tensor, n_fore: int, n_back: int, cutoff: float
):
    """Swap min-foreground and max-background components if their means cross."""
    min_fore = torch.min(mu[:n_fore, 1])
    max_back = torch.max(mu[:n_back, 0])
    min_fore_idx = int(torch.argmin(mu[:n_fore, 1]))
    max_back_idx = int(torch.argmax(mu[:n_back, 0]))

    mu[min_fore_idx, 1] = max_back
    mu[max_back_idx, 0] = min_fore

    n_fg = int((label == 1).sum())
    n_bg = int((label == 0).sum() - (data <= cutoff).sum())

    pi_fg = pi[min_fore_idx, 1].clone()
    pi_bg = pi[max_back_idx, 0].clone()

    # avoid zero-div
    n_fg = max(n_fg, 1)
    n_bg = max(n_bg, 1)

    pi[min_fore_idx, 1] = pi_bg * (n_bg / n_fg)
    pi[max_back_idx, 0] = pi_fg * (n_fg / n_bg)
    pi /= pi.sum(dim=0, keepdim=True) + _TINY

    sigma_fg = sigma[min_fore_idx, 1].clone()
    sigma_bg = sigma[max_back_idx, 0].clone()
    sigma[min_fore_idx, 1] = sigma_bg
    sigma[max_back_idx, 0] = sigma_fg
    return pi, mu, sigma


def _em_once(
    data_fore: torch.Tensor, data_back: torch.Tensor,
    n_fore: int, n_back: int,
    pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
    tol_: float = 1e-6, max_iter_: int = 30
):
    """Run a few EM steps to refresh GMM params for both classes."""
    loglh_old = None
    for _ in range(max_iter_):
        resp_bg = _calculate_resp(data_back, pi[:n_back, 0], mu[:n_back, 0], sigma[:n_back, 0])
        resp_fg = _calculate_resp(data_fore, pi[:n_fore, 1], mu[:n_fore, 1], sigma[:n_fore, 1])

        # M-step
        N_bg = resp_bg.sum(dim=0) + _TINY
        N_fg = resp_fg.sum(dim=0) + _TINY

        mu[:n_back, 0] = (1 / N_bg * (resp_bg * data_back.reshape(-1, 1))).sum(dim=0)
        mu[:n_fore, 1] = (1 / N_fg * (resp_fg * data_fore.reshape(-1, 1))).sum(dim=0)
        sigma[:n_back, 0] = torch.sqrt((1 / N_bg * (resp_bg * (data_back.reshape(-1, 1) - mu[:n_back, 0]) ** 2)).sum(dim=0)) + _TINY
        sigma[:n_fore, 1] = torch.sqrt((1 / N_fg * (resp_fg * (data_fore.reshape(-1, 1) - mu[:n_fore, 1]) ** 2)).sum(dim=0)) + _TINY

        pi[:n_back, 0] = N_bg / len(data_back)
        pi[:n_fore, 1] = N_fg / len(data_fore)

        # Monitor inner EM convergence (optional)
        loglh_bg = _loglh_gmm(data_back, pi[:n_back, 0], mu[:n_back, 0], sigma[:n_back, 0])
        loglh_fg = _loglh_gmm(data_fore, pi[:n_fore, 1], mu[:n_fore, 1], sigma[:n_fore, 1])
        loglh_new = loglh_bg + loglh_fg
        if loglh_old is None:
            loglh_old = loglh_new
            continue
        if torch.abs((loglh_new - loglh_old) / (loglh_new + _TINY)) < tol_:
            break
        loglh_old = loglh_new

    return pi, mu, sigma


# ---------------------
# Public API
# ---------------------

def segmentation(
    image: np.ndarray,
    frangi: np.ndarray,
    pixel_size: Tuple[float, float] | Tuple[float, float, float],
    beta1: float,
    beta2: float,
    cutoff: float,
    n_fore: int,
    n_back: int,
    max_iter: int,
    device: str | torch.device,
    *,
    progress: Optional[Callable[[int, float], None]] = None,
    tol: float = _DEFAULT_TOL,
) -> Tuple[np.ndarray, SegmentationInfo]:
    """
    Graphical model segmentation with GMM unary + pairwise smoothness.
    Returns (label_uint8, SegmentationInfo). If `progress` is provided, it is
    called every outer iteration with (iteration_index, delta_loglh).

    Parameters
    ----------
    image : np.ndarray
        2D/3D array; we expect C-first after pre-processing (Frangi already matched).
    frangi : np.ndarray
        Same shape as image; used for frangi potentials.
    pixel_size : (z, xy, ?) for 3D or (xy, ?) for 2D; only xy and optional z are used.
    device : str | torch.device
        "cpu", "cuda", "mps", etc.
    """
    dev = torch.device(device)
    data = torch.as_tensor(image, device=dev).unsqueeze(0)
    frg = torch.as_tensor(frangi, device=dev).unsqueeze(0)

    # normalize image to [0, 255] like the original
    data = (data - data.min()) / (data.max() - data.min() + _TINY) * 255.0

    n_component = max(n_fore, n_back)
    class_num = 2
    fpot0, fpot1 = _frangi_potential(frg, float(beta1))

    # Shape branches
    if data.ndim == 3:
        # (C,H,W)
        C, H, W = data.shape
        label = torch.randint(low=0, high=2, size=(C, H, W), device=dev)
        pixel_size_xy = float(pixel_size[0])
        pixel_size_z = None
        U_g = torch.zeros((class_num, H, W), device=dev)
    else:
        # (C,D,H,W)
        C, D, H, W = data.shape
        label = torch.randint(low=0, high=2, size=(C, D, H, W), device=dev)
        pixel_size_z, pixel_size_xy = float(pixel_size[0]), float(pixel_size[1])
        U_g = torch.zeros((class_num, D, H, W), device=dev)

    # GMM params
    pi = torch.zeros((n_component, class_num), device=dev)
    mu = torch.zeros((n_component, class_num), device=dev)
    sigma = torch.zeros((n_component, class_num), device=dev)

    # initial masking by cutoff
    label[data <= cutoff] = 0

    # one-time KMeans init then EM refresh each outer iter
    flag_switchable = True

    deltas: List[float] = []
    loglh_old_outer: Optional[torch.Tensor] = None

    for it in range(int(max_iter)):
        data_bg = data[label == 0]
        data_fg = data[label == 1]

        # safety: ensure non-empty sets
        if data_bg.numel() == 0 or data_fg.numel() == 0:
            # fallback: random split
            mask = (data > data.median()).squeeze()
            label[mask] = 1
            data_bg = data[label == 0]
            data_fg = data[label == 1]

        # init once
        if it == 0:
            pi, mu, sigma = _parameter_initialization(dev, pi, mu, sigma, data_fg, data_bg, n_fore, n_back)

        # EM on both classes
        pi, mu, sigma = _em_once(data_fg, data_bg, n_fore, n_back, pi, mu, sigma, tol_=1e-6, max_iter_=30)

        # swap if means cross
        if torch.min(mu[:n_fore, 1]) < torch.max(mu[:n_back, 0]):
            flag_switchable = True
            pi, mu, sigma = _switch_parameters(data, pi, mu, sigma, label, n_fore, n_back, float(cutoff))
        else:
            flag_switchable = False  # not used later, but kept for readability

        # update label by MAP
        if data.ndim == 3:
            U_c = _pairwise_potential(label, float(beta2), 0.1, fpot0, fpot1, dev, pixel_size_xy)
            U_g.zero_()
            U_g[0] = ((pi[:n_back, 0].reshape(-1, 1, 1, 1)
                      * _gaussian(data, mu[:n_back, 0].reshape(-1, 1, 1, 1), sigma[:n_back, 0].reshape(-1, 1, 1, 1)))
                      * torch.exp(-U_c[0])).sum(dim=0)
            U_g[1] = ((pi[:n_fore, 1].reshape(-1, 1, 1, 1)
                      * _gaussian(data, mu[:n_fore, 1].reshape(-1, 1, 1, 1), sigma[:n_fore, 1].reshape(-1, 1, 1, 1)))
                      * torch.exp(-U_c[1])).sum(dim=0)
            label = torch.argmax(U_g, dim=0).reshape((C, H, W)).to(dev)
        else:
            U_c = _pairwise_potential(label, float(beta2), 0.1, fpot0, fpot1, dev, pixel_size_xy, pixel_size_z)
            U_g.zero_()
            U_g[0] = ((pi[:n_back, 0].reshape(-1, 1, 1, 1, 1)
                      * _gaussian(data, mu[:n_back, 0].reshape(-1, 1, 1, 1, 1), sigma[:n_back, 0].reshape(-1, 1, 1, 1, 1)))
                      * torch.exp(-U_c[0])).sum(dim=0)
            U_g[1] = ((pi[:n_fore, 1].reshape(-1, 1, 1, 1, 1)
                      * _gaussian(data, mu[:n_fore, 1].reshape(-1, 1, 1, 1, 1), sigma[:n_fore, 1].reshape(-1, 1, 1, 1, 1)))
                      * torch.exp(-U_c[1])).sum(dim=0)
            label = torch.argmax(U_g, dim=0).reshape((C, D, H, W)).to(dev)

        # enforce cutoff again
        label[data <= cutoff] = 0

        # outer log-likelihood monitor
        loglh_new_outer = torch.log(torch.where(label == 0, U_g[0], U_g[1]) + _TINY).sum()

        if loglh_old_outer is not None:
            delta = torch.abs((loglh_new_outer - loglh_old_outer) / (loglh_new_outer + _TINY))
            deltas.append(float(delta.detach().cpu()))
            if progress is not None:
                # iteration index and ΔlogL (relative)
                progress(it, float(delta.detach().cpu()))
            if delta < tol:
                info = SegmentationInfo(iterations_run=it + 1, deltas=deltas, converged=True)
                out = (label * 255).detach().cpu().numpy().astype(np.uint8)
                return out, info
        else:
            # first iteration -> report delta as NaN for UI consistency
            if progress is not None:
                progress(it, float("nan"))

        loglh_old_outer = loglh_new_outer

    # not converged within max_iter
    info = SegmentationInfo(iterations_run=int(max_iter), deltas=deltas, converged=False)
    out = (label * 255).detach().cpu().numpy()
    return out, info
