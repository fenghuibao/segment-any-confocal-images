#!/usr/bin/env python3
# Author: Huibao Feng
# Date: 2025-11-02

import torch
from torch import nn
from .gaussian_smoothing import GaussianSmoothing
from itertools import combinations_with_replacement
import numpy as np


class FrangiFilter(nn.Module):

    def __init__(self, channels, kernel_size, sigmas, dim, device='cpu', zx_ratio=1, alpha=0.5, beta=0.5, gamma=2):
        """
        Arguments:
            channels (int, sequence): Number of channels of the input tensors. Output will
                have this number of channels as well.
            kernel_size (int, sequence): Size of the gaussian kernel.
            sigmas (list, sequence): List of standard deviations of the gaussian kernels.
            beta (float, sequence): Beta parameter of Frangi filter.
            beta (с, sequence): С parameter of Frangi filter.
                Default value is 2 (spatial).
        """
        super(FrangiFilter, self).__init__()
        self.channels = channels
        self.sigmas = sigmas
        self.kernel_size= kernel_size
        self.dim = dim
        self.device = device
        self.alpha=alpha
        self.beta = beta
        self.gamma = gamma
        self.zx_ratio = zx_ratio
        
    
    def Hessian_matrix(self, image, sigma, zx_ratio=1.0):
        coef = 1/np.sqrt(2)
        if self.dim == 2:
            Gx = GaussianSmoothing(channels=self.channels, kernel_size=self.kernel_size, sigma=coef*sigma, dim=2, order='x', device=self.device)
            Gy = GaussianSmoothing(channels=self.channels, kernel_size=self.kernel_size, sigma=coef*sigma, dim=2, order='y', device=self.device)
            Hxx = Gx(Gx(image))
            Hxy = Gx(Gy(image))
            Hyy = Gy(Gy(image))
            return Hyy, Hxy, Hxx
        else:
            Gx = GaussianSmoothing(channels=self.channels, kernel_size=self.kernel_size, sigma=coef*sigma, dim=3, order='x', device=self.device)
            Gy = GaussianSmoothing(channels=self.channels, kernel_size=self.kernel_size, sigma=coef*sigma, dim=3, order='y', device=self.device)
            Gz = GaussianSmoothing(channels=self.channels, kernel_size=self.kernel_size, sigma=coef*sigma*zx_ratio, dim=3, order='z', device=self.device)
            Hzz = Gz(Gz(image))
            Hzy = Gz(Gy(image))
            Hzx = Gz(Gx(image))
            Hyy = Gy(Gy(image))
            Hyx = Gy(Gx(image))
            Hxx = Gx(Gx(image))
            return Hzz, Hzy, Hzx, Hyy, Hyx, Hxx        
        

    def _symmetric_image(self, S_elems):
        image = S_elems[0]
        symmetric_image = torch.zeros(
            image.shape + (image.ndim-1, image.ndim-1), dtype=S_elems[0].dtype
        ).to(self.device)
        for idx, (row, col) in enumerate(
            combinations_with_replacement(range(image.ndim-1), 2)
        ):
            symmetric_image[..., row, col] = S_elems[idx]
            symmetric_image[..., col, row] = S_elems[idx]
        return symmetric_image
    
    def _symmetric_compute_eigenvalues(self, S_elems):
        if len(S_elems) == 3:
            M00, M01, M11 = S_elems
            mean = (M00 + M11) * 0.5
            hsqrtdet = torch.hypot(M01, (M00 - M11) * 0.5)  # sqrt(x^2 + y^2) elementwise
            eigs = torch.stack([mean + hsqrtdet, mean - hsqrtdet], dim=0)
        else:
            matrices = self._symmetric_image(S_elems)
            matrices = matrices.to('cpu')
            eigs = torch.flip(torch.linalg.eigvalsh(matrices), dims=[-1])
            eigs = eigs.movedim(-1, 0)
            eigs = eigs.to(self.device)
        return torch.squeeze(eigs)

    def _calc_frangi_response(self, image):
        filtered_max = torch.zeros_like(image)
        eps = torch.tensor(1e-15, dtype=torch.float32, device=self.device)
        for sigma in self.sigmas:
            eigvals = self._symmetric_compute_eigenvalues(
                self.Hessian_matrix(
                    image, sigma, self.zx_ratio
                )
            )
            sort_idx = eigvals.abs().argsort(dim=0)
            eigvals = torch.gather(eigvals, 0, sort_idx)
            lambda1 = eigvals[0]
            if self.dim == 2:
                # take max with eps for all remaining eigenvalues (only one: lambda2)
                lambda2 = torch.maximum(eigvals[1], eps)
                r_a = torch.tensor(float("inf"), dtype=torch.float32, device=self.device) 
                r_b = lambda1.abs() / lambda2  
            else:  # ndim == 3
                lambda2 = torch.maximum(eigvals[1], eps)
                lambda3 = torch.maximum(eigvals[2], eps)
                r_a = lambda2 / lambda3 
                r_b = lambda1.abs() / torch.sqrt(lambda2 * lambda3) 

            s = torch.sqrt((eigvals**2).sum(dim=0)) 
            vals = 1.0 - torch.exp(-(r_a**2) / (2 * (self.alpha**2)))             # plate sensitivity
            vals = vals * torch.exp(-(r_b**2) / (2 * (self.beta**2)))              # blobness
            vals = vals * (1.0 - torch.exp(-(s**2) / (2 * (self.gamma**2))))   # structuredness
            filtered_max = torch.maximum(filtered_max, vals)
            filtered_max = torch.nan_to_num(filtered_max , nan=0.0)

        return filtered_max
    

    def forward(self, image):
        """
        Apply Frangi filter on a batch of images.
        Arguments:
            img (torch.Tensor, sequence): Tensor of shape (bs, channels, h, w)
        """
        image = torch.tensor(image, dtype=torch.float32).to(self.device)
        frangi_resp = self._calc_frangi_response(image)
        return frangi_resp
