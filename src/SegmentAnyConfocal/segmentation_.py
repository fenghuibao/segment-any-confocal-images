import numpy as np 
import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import pickle
import torch.nn.functional as F
import numpy as np

from aicspylibczi import CziFile
from aicsimageio.readers import CziReader
import time
#from tvtk.util import ctf
import os
import cv2
#from frangi_filter.frangi_filter import *

device = 'mps'
tol = 1e-8


# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--czifile', help='input czi file', required=True)
# parser.add_argument('-b1', '--beta1', type=float, help='appearance factor', default=1)
# parser.add_argument('-b2', '--beta2', type=float, help='smoothness factor', default=1)
# parser.add_argument('-cutoff', '--cutoff', type=float, help='cutoff', default=0)
# parser.add_argument('-nfore', '--nforeground', type=int, help='foreground', default=10)
# parser.add_argument('-nback', '--nbackground', type=int, help='background', default=2)
# parser.add_argument('-maxiter', '--maxiter', help='maxiter', default=200)
# parser.add_argument('-c', '--channel', type=int, help='channel', required=True)
# args = parser.parse_args()

# czifile = args.czifile
# beta1 = args.beta1
# beta2 = args.beta2
# cutoff = args.cutoff
# n_fore = args.nforeground
# n_back = args.nbackground
# max_iter = args.maxiter
# channel = args.channel

def Gaussian(x, mu, sigma):
    return 1 / torch.sqrt(2 * torch.pi * sigma ** 2) * torch.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def calculate_resp(X, pi, mu, sigma):
	resp = pi * Gaussian(X.reshape(-1, 1), mu, sigma)
	return resp / (resp.sum(axis=1).reshape(-1, 1) + 1e-15)

def loglh_GMM(X, pi, mu, sigma):
	return torch.log((pi * Gaussian(X.reshape(-1, 1), mu, sigma)).sum(axis=1)).sum(axis=0)
    

def get_adjacent_labels(image_label):
    if image_label.ndim == 3:
        left = F.pad(image_label, (1, 0, 0, 0))
        left = left[:, :, :-1]
        right = F.pad(image_label, (0, 1, 0, 0))
        right = right[:, :, 1:]
        front = F.pad(image_label, (0, 0, 1, 0))
        front = front[:, :-1, :]
        back = F.pad(image_label, (0, 0, 0, 1))
        back = back[:, 1:, :]
        return left, right, front, back
    else:
        left = F.pad(image_label, (1, 0, 0, 0, 0, 0))
        left = left[:, :, :, :-1]
        right = F.pad(image_label, (0, 1, 0, 0, 0, 0))
        right = right[:, :, :, 1:]
        front = F.pad(image_label, (0, 0, 1, 0, 0, 0))
        front = front[:, :, :-1, :]
        back = F.pad(image_label, (0, 0, 0, 1, 0, 0))
        back = back[:, :, 1:, :]
        up = F.pad(image_label, (0, 0, 0, 0, 1, 0))
        up = up[:, :-1, :, :]
        down = F.pad(image_label, (0, 0, 0, 0, 0, 1))
        down = down[:, 1:, :, :]
        return left, right, front, back, up, down

def smoothness_potential(label, smoothness_energy):
    indicator_0 = torch.where(label == 0, -1, 1)
    indicator_1 = torch.where(label == 1, -1, 1)
    potential_0 = indicator_0 * smoothness_energy
    potential_1 = indicator_1 * smoothness_energy
    return potential_0, potential_1

def frangi_potential(frangi, beta1):
    frangi_0 = frangi / frangi.max()
    frangi_1 = 1 - frangi_0
    frangi_potential_0 = beta1 * torch.log(frangi_0 + 1e-15)
    frangi_potential_1 = beta1 * torch.log(frangi_1 + 1e-15)
    return frangi_potential_0, frangi_potential_1

def pairwise_potential(image_label, beta2, sigma, frangi_potential_0, frangi_potential_1, device, pixel_size_xy, pixel_size_z=None):
    smoothness_xy = beta2 * np.exp(-pixel_size_xy ** 2 / 2 / sigma ** 2)
    if pixel_size_z:
        smoothness_z = beta2 * np.exp(-pixel_size_z ** 2 / 2 / sigma ** 2)
    if image_label.ndim == 3:
        left, right, front, back = get_adjacent_labels(image_label)
        potential_left_0, potential_left_1 = smoothness_potential(left, smoothness_xy)
        potential_right_0, potential_right_1 = smoothness_potential(right, smoothness_xy)
        potential_front_0, potential_front_1 = smoothness_potential(front, smoothness_xy)
        potential_back_0, potential_back_1 = smoothness_potential(back, smoothness_xy)
        potential_0 = potential_left_0 + potential_right_0 + potential_front_0 + potential_back_0 + frangi_potential_0
        potential_1 = potential_left_1 + potential_right_1 + potential_front_1 + potential_back_1 + frangi_potential_1
    else:
        left, right, front, back, up, down = get_adjacent_labels(image_label)
        potential_left_0, potential_left_1 = smoothness_potential(left, smoothness_xy)
        potential_right_0, potential_right_1 = smoothness_potential(right, smoothness_xy)
        potential_front_0, potential_front_1 = smoothness_potential(front, smoothness_xy)
        potential_back_0, potential_back_1 = smoothness_potential(back, smoothness_xy)
        potential_up_0, potential_up_1 = smoothness_potential(up, smoothness_z)
        potential_down_0, potential_down_1 = smoothness_potential(down, smoothness_z)
        potential_0 = potential_left_0 + potential_right_0 + potential_front_0 + potential_back_0 + potential_up_0 + potential_down_0 + frangi_potential_0
        potential_1 = potential_left_1 + potential_right_1 + potential_front_1 + potential_back_1 + potential_up_1 + potential_down_1 + frangi_potential_1
    
    potential = torch.concat([potential_0, potential_1]).to(device)
    return potential

def parameter_initialization(pi, mu, sigma, data_foreground, data_background, n_fore, n_back):
    mu[:n_back, 0] = torch.tensor(KMeans(n_clusters=n_back, n_init='auto').fit(data_background.cpu().reshape(-1, 1)).cluster_centers_.reshape(-1), dtype=torch.float32).to(device)
    mu[:n_fore, 1] = torch.tensor(KMeans(n_clusters=n_fore, n_init='auto').fit(data_foreground.cpu().reshape(-1, 1)).cluster_centers_.reshape(-1), dtype=torch.float32).to(device)
    sigma[:n_back, 0] = 256
    sigma[:n_fore, 1] = 256
    pi[:n_back, 0] = 1 / n_back
    pi[:n_fore, 1] = 1 / n_fore
    return pi, mu, sigma

def switch_parameters(data, pi, mu, sigma, label, n_fore, n_back, cutoff):
    min_foreground = min(mu[:n_fore, 1])
    max_background = max(mu[:n_back, 0])
    min_foreground_index = mu[:, 1].tolist().index(min(mu[:n_fore, 1]))
    max_background_index = mu[:, 0].tolist().index(max(mu[:n_back, 0]))
    mu[min_foreground_index, 1] = max_background
    mu[max_background_index, 0] = min_foreground 

    n_foreground = len(label[label == 1])
    n_background = len(label[label == 0]) - len(label[data <= cutoff])

    pi_foreground = pi[min_foreground_index, 1]
    pi_background = pi[max_background_index, 0]

    pi[min_foreground_index, 1] = pi_background * n_background / n_foreground
    pi[max_background_index, 0] = pi_foreground * n_foreground / n_background
    pi = pi / pi.sum(axis = 0)
    
    sigma_foreground = sigma[min_foreground_index, 1]
    sigma_background = sigma[max_background_index, 0]
    sigma[min_foreground_index, 1] = sigma_background
    sigma[max_background_index, 0] = sigma_foreground
    return pi, mu, sigma

def EM(data_foreground, data_background, n_fore, n_back, pi, mu, sigma, tol_, max_iter_):

    for iter in range(max_iter_):
        # expectation
        resp_background = calculate_resp(data_background, pi[:n_back, 0], mu[:n_back, 0], sigma[:n_back, 0])
        resp_foreground = calculate_resp(data_foreground, pi[:n_fore, 1], mu[:n_fore, 1], sigma[:n_fore, 1])

        # maximization
        N_background = resp_background.sum(axis=0) + 1e-15
        N_foreground = resp_foreground.sum(axis=0) + 1e-15
        mu[:n_back, 0] = (1 / N_background * (resp_background * data_background.reshape(-1, 1))).sum(axis=0)
        mu[:n_fore, 1] = (1 / N_foreground * (resp_foreground * data_foreground.reshape(-1, 1))).sum(axis=0)
        sigma[:n_back, 0] = torch.sqrt((1 / N_background * (resp_background * (data_background.reshape(-1, 1) - mu[:n_back, 0]) ** 2)).sum(axis=0)) + 1e-15
        sigma[:n_fore, 1] = torch.sqrt((1 / N_foreground * (resp_foreground * (data_foreground.reshape(-1, 1) - mu[:n_fore, 1]) ** 2)).sum(axis=0)) + 1e-15
        pi[:n_back, 0] = N_background / len(data_background)
        pi[:n_fore, 1] = N_foreground / len(data_foreground)

        # log-likelihood
        loglh_background = loglh_GMM(data_background, pi[:n_back, 0], mu[:n_back, 0], sigma[:n_back, 0])
        loglh_foreground = loglh_GMM(data_foreground, pi[:n_fore, 1], mu[:n_fore, 1], sigma[:n_fore, 1])
        loglh_new_ = loglh_background + loglh_foreground
        if iter == 0:
            loglh_old_ = loglh_new_
            continue
        else:
            delta_loglh_ = torch.abs((loglh_new_ - loglh_old_) / loglh_new_)
            loglh_old_ = loglh_new_
        if delta_loglh_ < tol_:
            break
    return pi, mu, sigma

def segmentation(image, frangi, pixel_size, beta1, beta2, cutoff, n_fore, n_back, max_iter, device):
    print(pixel_size)
    data = torch.tensor(image).unsqueeze(0).to(device)
    data = (data - data.min()) / (data.max() - data.min()) * 255
    frangi = torch.tensor(frangi).unsqueeze(0).to(device)
    n_component = max(n_fore, n_back)
    class_num = 2
    frangi_potential_0, frangi_potential_1 = frangi_potential(frangi, beta1)

    if data.ndim == 3:
        C, H, W = data.shape
        label = torch.randint(low=0, high=2, size=(C, H, W), device=device)
        U_c = torch.zeros((class_num, H, W))
        pixel_size_xy, _ = pixel_size
    else:
        C, D, H, W = data.shape
        label = torch.randint(low=0, high=2, size=(C, D, H, W), device=device)
        U_c = torch.zeros((class_num, D, H, W))
        pixel_size_z, pixel_size_xy, _ = pixel_size
    
    pi = torch.zeros((n_component, class_num)).to(device)
    mu = torch.zeros((n_component, class_num)).to(device)
    sigma = torch.zeros((n_component, class_num)).to(device)
    label[data <= cutoff] = 0
    flag = True

    for iter in range(max_iter):
        data_background = data[label == 0]
        data_foreground = data[label == 1]
        if flag:
            pi, mu, sigma = parameter_initialization(pi, mu, sigma, data_foreground, data_background, n_fore, n_back)
        # expectation-maximization
        pi, mu, sigma = EM(data_foreground, data_background, n_fore, n_back, pi, mu, sigma, tol_=1e-6, max_iter_=30)

        if min(mu[:n_fore, 1]) < max(mu[:n_back, 0]):
            flag = True
            pi, mu, sigma = switch_parameters(data, pi, mu, sigma, label, n_fore, n_back, cutoff)
        else: 
            flag = False
        
        if data.ndim == 3:
            U_c = pairwise_potential(label, beta2, 0.1, frangi_potential_0, frangi_potential_1, device, pixel_size_xy)
            U_g = torch.zeros((class_num, H, W)).to(device)
            U_g[0] = ((pi[:n_back, 0].reshape(-1, 1, 1, 1) * Gaussian(data, mu[:n_back, 0].reshape(-1, 1, 1, 1), sigma[:n_back, 0].reshape(-1, 1, 1, 1))) * torch.exp(-U_c[0])).sum(axis=0)
            U_g[1] = ((pi[:n_fore, 1].reshape(-1, 1, 1, 1) * Gaussian(data, mu[:n_fore, 1].reshape(-1, 1, 1, 1), sigma[:n_fore, 1].reshape(-1, 1, 1, 1))) * torch.exp(-U_c[1])).sum(axis=0)
            label = torch.argmax(U_g, axis=0).reshape((C, H, W)).to(device)
        else:
            U_c = pairwise_potential(label, beta2, 0.1, frangi_potential_0, frangi_potential_1, device, pixel_size_xy, pixel_size_z)
            U_g = torch.zeros((class_num, D, H, W)).to(device)
            U_g[0] = ((pi[:n_back, 0].reshape(-1, 1, 1, 1, 1) * Gaussian(data, mu[:n_back, 0].reshape(-1, 1, 1, 1, 1), sigma[:n_back, 0].reshape(-1, 1, 1, 1, 1))) * torch.exp(-U_c[0])).sum(axis=0)
            U_g[1] = ((pi[:n_fore, 1].reshape(-1, 1, 1, 1, 1) * Gaussian(data, mu[:n_fore, 1].reshape(-1, 1, 1, 1, 1), sigma[:n_fore, 1].reshape(-1, 1, 1, 1, 1))) * torch.exp(-U_c[1])).sum(axis=0)
            label = torch.argmax(U_g, axis=0).reshape((C, D, H, W)).to(device)
        
        label[data <= cutoff] = 0
        loglh_new = torch.log(torch.where(label==0, U_g[0], U_g[1])).sum()

        if iter == 0:
            loglh_old = loglh_new
            continue
        else:
            delta_loglh = torch.abs((loglh_new - loglh_old) / loglh_new)
            loglh_old = loglh_new
        if delta_loglh < tol:
            break

        print(iter)
    return label * 255

    # with open('%s_label_%s_%d.pickle'%(czifile.split('.')[0],stack, channel), 'wb') as f:
    #     pickle.dump(label.cpu().numpy(), f)
            
    
# train(
#             czifile,
#             beta1,
#             beta2,
#             n_fore,
#             n_back,
#             max_iter,
#             channel,
#             tol=1e-8
#         )


##pixel size z