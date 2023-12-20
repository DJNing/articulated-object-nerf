# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------


import numpy as np
import torch
import torch.nn.functional as F


def load_state_dict_and_report(model, ckpt_file):
    # Load the checkpoint file
    checkpoint = torch.load(ckpt_file)  # Use 'map_location' to load on CPU if needed

    # Get the state_dict from the checkpoint
    pretrained_state_dict = checkpoint['state_dict']

    # Get the model's state_dict
    model_state_dict = model.state_dict()

    # Initialize lists to store missing and unexpected keys
    missing_keys = []
    unexpected_keys = []

    # Iterate through the keys in the pretrained_state_dict
    for key, value in pretrained_state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
            else:
                print(f"Size mismatch for key '{key}': expected {model_state_dict[key].shape}, but got {value.shape}")
        else:
            missing_keys.append(key)

    # Check for unexpected keys
    for key in model_state_dict.keys():
        if key not in pretrained_state_dict:
            unexpected_keys.append(key)

    # Load the modified state_dict into the model
    model.load_state_dict(model_state_dict)

    # Report missing and unexpected keys
    if missing_keys:
        print("Missing keys in model state_dict:")
        for key in missing_keys:
            print(key)

    if unexpected_keys:
        print("Unexpected keys in pretrained state_dict:")
        for key in unexpected_keys:
            print(key)

def img2mse(x, y):
    return torch.mean((x - y) ** 2)

def img2mse_weighted(x, y, prob):
    diff = (x - y) ** 2
    result = prob * diff
    loss = torch.mean(result)
    return loss

def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10)


def cast_rays(t_vals, origins, directions):
    return origins[..., None, :] + t_vals.unsqueeze(-1) * directions[..., None, :]


def get_ray_lim(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length=2):
    batch_near, batch_far = get_ray_lim_box(
        rays_o, rays_d, box_side_length=box_side_length
    )
    is_ray_valid = batch_far > batch_near
    if torch.any(is_ray_valid).item():
        batch_near[~is_ray_valid] = batch_near[is_ray_valid].min()
        batch_far[~is_ray_valid] = batch_far[is_ray_valid].max()
    batch_near[batch_near < 0] = 0
    batch_far[batch_far < 0] = 0
    return batch_near, batch_far


def get_ray_lim_box(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)
    bb_min = [
        -1 * (box_side_length / 2),
        -1 * (box_side_length / 2),
        -1 * (box_side_length / 2),
    ]
    bb_max = [
        1 * (box_side_length / 2),
        1 * (box_side_length / 2),
        1 * (box_side_length / 2),
    ]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)
    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()
    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[
        ..., 0
    ]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[
        ..., 0
    ]
    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[
        ..., 1
    ]
    tymax = (
        bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]
    ) * invdir[..., 1]
    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False
    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)
    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[
        ..., 2
    ]
    tzmax = (
        bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]
    ) * invdir[..., 2]
    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False
    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)
    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2
    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)



def sample_along_rays(
    rays_o,
    rays_d,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
):
    bsz = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    coords = cast_rays(t_vals, rays_o, rays_d)

    return t_vals, coords


def pos_enc(x, min_deg, max_deg):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else:  # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd, nocs=None, seg=None):
    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-density[..., 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[..., :1]),
            torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )

    weights = alpha * accum_prod

    comp_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)

    depth = torch.nan_to_num(depth, float("inf"))
    depth = torch.clamp(depth, torch.min(depth), torch.max(depth))

    acc = weights.sum(dim=-1)
    inv_eps = 1 / eps

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc.unsqueeze(-1))

    if nocs is not None:
        comp_nocs = (weights.unsqueeze(-1) * nocs).sum(dim=-2)
        return comp_rgb, acc, weights, comp_nocs
    else:
        return comp_rgb, acc, weights, depth

def filter_seg_from_acc(seg, acc):
    """
    Select rows from tensor a where b is non-zero.
    
    Args:
        seg (torch.Tensor): Input tensor of shape [N, K].
        acc (torch.Tensor): Input tensor of shape [N, 1].
        
    Returns:
        torch.Tensor: Result tensor of shape [s, K], where s is the number of non-zero values in acc.
    """
    # Find indices where b is non-zero
    non_zero_indices = torch.nonzero(acc.view(-1)).squeeze()
    
    # Select rows from tensor a where b is non-zero
    result = seg[non_zero_indices]
    
    return result
def volumetric_part_rendering(rgb, density, t_vals, dirs, white_bkgd, seg, part_idx=0):
    '''
    [rays, n_samples, channle]
    '''
    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    def get_weights(density, dists, eps=1e-10):
        alpha = 1.0 - torch.exp(-density[..., 0] * dists)
        accum_prod = torch.cat(
            [
                torch.ones_like(alpha[..., :1]),
                torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
            ],
            dim=-1,
        )

        weights = alpha * accum_prod
        return weights
    # get seg_density
    seg_fg = seg[:, :, part_idx:part_idx+1]
    
    seg_density = seg_fg * density
    weights = get_weights(density, dists, eps=eps)
    seg_weights = get_weights(seg_density, dists, eps=eps)
    # seg_weights = get_weights(seg, dists, eps=eps)

    comp_rgb = (seg_weights.unsqueeze(-1) * rgb).sum(dim=-2)

    # comp_rgb_seg = (weights.unsqueeze(-1) * seg_weights.unsqueeze(-1) * rgb).sum(dim=-2)

    if torch.isnan(comp_rgb).any():
        print('nan in rgb')

    depth = (seg_weights * t_vals).sum(dim=-1)

    depth = torch.nan_to_num(depth, float("inf"))
    depth = torch.clamp(depth, torch.min(depth), torch.max(depth))

    acc = weights.sum(dim=-1)
    seg_opacity = seg_weights.sum(dim=-1)
    # opacity = (weights * seg_weights).sum(dim=-1)
    inv_eps = 1 / eps

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc.unsqueeze(-1))

    
    comp_nocs = None

    # also render non-base(nb) part for negative label
    nb_seg, _ = seg[:, :, 1:].max(dim=-1, keepdim=True)
    nb_density = nb_seg * density
    nb_weights = get_weights(nb_density, dists, eps=eps)
    nb_opacity = nb_weights.sum(dim=-1)
    ret_dict = {
        'comp_rgb': comp_rgb,
        'acc': acc,
        'seg_weights': seg_weights,
        'seg_opa': seg_opacity,
        'weights': weights,
        'depth': depth,
        'comp_nocs': comp_nocs,
        'nb_opa': nb_opacity
    }
    return ret_dict

def volumetric_seg_rendering(rgb, density, t_vals, dirs, white_bkgd, seg, nocs=None):
    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    def get_weights(density, dists, eps=1e-10):
        alpha = 1.0 - torch.exp(-density[..., 0] * dists)
        accum_prod = torch.cat(
            [
                torch.ones_like(alpha[..., :1]),
                torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
            ],
            dim=-1,
        )

        weights = alpha * accum_prod
        return weights
    
    weights = get_weights(density, dists, eps=eps)
    seg_weights = get_weights(seg, dists, eps=eps)

    comp_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=-2)

    comp_rgb_seg = (weights.unsqueeze(-1) * seg_weights.unsqueeze(-1) * rgb).sum(dim=-2)

    if torch.isnan(comp_rgb).any():
        print('nan in rgb')

    depth = (weights * t_vals).sum(dim=-1)

    depth = torch.nan_to_num(depth, float("inf"))
    depth = torch.clamp(depth, torch.min(depth), torch.max(depth))

    acc = weights.sum(dim=-1)
    opacity = (weights * seg_weights).sum(dim=-1)
    inv_eps = 1 / eps

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc.unsqueeze(-1))

    if nocs is not None:
        comp_nocs = (weights.unsqueeze(-1) * nocs).sum(dim=-2)
        # return comp_rgb, acc, weights, comp_nocs
    else:
        # return comp_rgb, acc, weights, depth
        comp_nocs = None


    ret_dict = {
        'comp_rgb': comp_rgb,
        'comp_rgb_seg': comp_rgb_seg,
        'acc': acc,
        'weights': weights,
        'depth': depth,
        'comp_nocs': comp_nocs,
        'comp_seg':seg_weights.sum(dim=-1),
        'opacity': opacity,
    }
    return ret_dict

def get_weights(density, dists, eps=1e-10):
    alpha, accum_prod = get_coeff(density, dists, eps=eps)

    weights = alpha * accum_prod
    return weights

def get_coeff(density, dists, eps=1e-10):
    alpha = 1.0 - torch.exp(-density[..., 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[..., :1]),
            torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )
    return alpha, accum_prod

def volumetric_composite_rendering(rgb, density, t_vals, dirs, \
                                   seg, rgb_activation=False):
    '''
    rgb: [part_num, ray_num, sample_num, 3]
    density: [part_num, ray_num, sample_num, 1]
    seg: [part_num, ray_num, sample_num, part_num (include_bg: +1)]
    t_vals: [part_num * ray_num, sample_num]
    '''
    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1) 
    # dists = dists.unsqueeze(-1) # [r, s, 1]
    dists = dists.unsqueeze(1) # [r*p, 1, s]
    dists = dists.view(rgb.shape[0], rgb.shape[1], rgb.shape[2])

    # # canonical rendering
    # ca_rgb = rgb[0, :, :, :]
    # ca_dists = dists[0, :, :]
    # ca_density = density[0, :, :, :]




    part_num, ray_num, sample_num, _ = rgb.shape
    rgb = rgb.permute(1, 0, 2, 3) # [r, p, s, c]
    density = density.permute(1, 0, 2, 3) # [r, p, s, 1]
    seg = seg.permute(1, 0, 2, 3) # [r, p, s, p]
    seg_fg = torch.cat([seg[:, i:i+1, :, i:i+1] for i in range(seg.shape[1])], dim=1) # [r, p, s, 1]
    seg_density = seg_fg * density
    dists = dists.permute(1, 0, 2)
    lambda_i = seg_density[..., 0] * dists # [ r, p, s]
    alpha = 1.0 - torch.exp(-lambda_i) # [r, p, s]
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[..., :1]),
            torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )
    T_i = accum_prod.prod(dim=1) # [r, s]
    T_i = T_i.unsqueeze(1).repeat(1, accum_prod.shape[1], 1) # [r, p, s]
    rgb_alpha = alpha.unsqueeze(dim=-1) * rgb # [r, p, s, 3]
    rgb_part = T_i.unsqueeze(dim=-1) * rgb_alpha #[r, p, s, 3]
    rgb_part = rgb_part.sum(dim=2) # [r, p, 3]

    seg_alpha = alpha.unsqueeze(dim=-1) * seg # [r, p, s, p]
    seg_part = T_i.unsqueeze(dim=-1) * seg_alpha
    seg_part = seg_part.sum(dim=2)
    seg_final = seg_part.sum(dim=1)
    if rgb_activation:
        import torch.nn as nn
        rgb_act = nn.Softmax(dim=1)
        rgb_weight = rgb_part.sum(dim=-1) # [r, p]
        
        from models.vanilla_nerf.model_nerfseg import OneHotActivation
        one_hot_act = OneHotActivation
        rgb_mask = one_hot_act(rgb_act(rgb_weight))
        rgb_part_act = rgb_mask.unsqueeze(-1) * rgb_part # [r, p, 3]
        rgb_final = rgb_part_act.sum(dim=1)
        pass
    else:
        rgb_final = rgb_part.sum(dim=1)
    opacity = (T_i * alpha).sum(dim=1).sum(dim=1)
    depth = (T_i * (alpha * t_vals.view(rgb.shape[1], rgb.shape[0], -1).permute(1, 0, 2))).sum(dim=1).sum(1)
    opa_acc = (T_i * alpha).sum(dim=-1)
    # opa_prob = s
    # weights = (T_i * alpha).view(-1, T_i.shape[-1])

    weights = (T_i * alpha).sum(dim=1).repeat(part_num, 1)
    # weights = weights.repeat(part_num, 1)
    ret_dict = {
        "comp_rgb": rgb_final,
        "opacity": opacity,
        "depth": depth,
        "weights": weights,
        "opa_part": opa_acc,
        "comp_seg": seg_final,
        "canonical_rgb": None,
        "canonical_opa": None
    }
    return ret_dict

def volumetric_rendering_seg_mask(rgb, density, t_vals, dirs, white_bkgd, seg_mask, seg, nocs=None):
    '''
    seg: pre-select the correponding idx before feeding in
    '''
    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    
    
    mask_density = density * seg_mask

    weights = get_weights(mask_density, dists, eps=eps)

    comp_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=-2)

    comp_seg = (weights.unsqueeze(-1) * seg).sum(dim=-2)
    if torch.isnan(comp_rgb).any():
        print('nan in rgb')

    depth = (weights * t_vals).sum(dim=-1)

    depth = torch.nan_to_num(depth, float("inf"))
    depth = torch.clamp(depth, torch.min(depth), torch.max(depth))

    acc = weights.sum(dim=-1)
    inv_eps = 1 / eps

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc.unsqueeze(-1))

    if nocs is not None:
        comp_nocs = (weights.unsqueeze(-1) * nocs).sum(dim=-2)
        # return comp_rgb, acc, weights, comp_nocs
    else:
        # return comp_rgb, acc, weights, depth
        comp_nocs = None


    ret_dict = {
        'comp_rgb': comp_rgb,
        'comp_rgb_seg': comp_rgb,
        'acc': acc,
        'weights': weights,
        'depth': depth,
        'comp_nocs': comp_nocs,
        'comp_seg':comp_seg,
        'opacity': acc,
    }
    return ret_dict


def volumetric_rendering_with_seg(rgb, density, t_vals, dirs, white_bkgd, seg, nocs=None):
    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    alpha = 1.0 - torch.exp(-density[..., 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[..., :1]),
            torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )

    weights = alpha * accum_prod

    comp_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=-2)

    comp_seg = (weights.unsqueeze(-1) * seg).sum(dim=-2)

    # mask rgb with rendered seg


    if torch.isnan(comp_rgb).any():
        print('nan in rgb')

    depth = (weights * t_vals).sum(dim=-1)

    depth = torch.nan_to_num(depth, float("inf"))
    depth = torch.clamp(depth, torch.min(depth), torch.max(depth))

    acc = weights.sum(dim=-1)
    inv_eps = 1 / eps

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc.unsqueeze(-1))

    if nocs is not None:
        comp_nocs = (weights.unsqueeze(-1) * nocs).sum(dim=-2)
        # return comp_rgb, acc, weights, comp_nocs
    else:
        # return comp_rgb, acc, weights, depth
        comp_nocs = None


    ret_dict = {
        'comp_rgb': comp_rgb,
        'acc': acc,
        'weights': weights,
        'depth': depth,
        'comp_nocs': comp_nocs,
        'comp_seg':comp_seg
    }
    return ret_dict

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def sorted_piecewise_constant_pdf(
    bins, weights, num_samples, randomized, float_min_eps=2**-32
):
    """
    bins: tensor = [N, 64]
    weights: tensor = [N, 63]
    num_samples: int = 128
    
    """
    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights = weights + padding / weights.shape[-1]
    weight_sum = weight_sum + padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], dim=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        dim=-1,
    )

    s = 1 / num_samples
    if randomized:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u.unsqueeze(1) >= cdf.unsqueeze(-1)

    bin0 = (mask * bins.unsqueeze(-1) + ~mask * bins[:, :1].unsqueeze(-1)).max(dim=-2)[0] # lower bound?
    bin1 = (~mask * bins.unsqueeze(-1) + mask * bins[:, -1:].unsqueeze(-1)).min(dim=-2)[0] # upper bound?
    # Debug Here
    cdf0 = (mask * cdf.unsqueeze(-1) + ~mask * cdf[..., :1].unsqueeze(-1)).max(dim=-2)[0]
    cdf1 = (~mask * cdf.unsqueeze(-1) + mask * cdf[..., -1:].unsqueeze(-1)).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples


def sample_pdf(bins, weights, origins, directions, t_vals, num_samples, randomized):
    t_samples = sorted_piecewise_constant_pdf(
        bins, weights, num_samples, randomized
    ).detach()
    t_vals = torch.sort(torch.cat([t_vals, t_samples], dim=-1), dim=-1).values
    coords = cast_rays(t_vals, origins, directions)
    return t_vals, coords


def generate_samples(x_lim, y_lim, z_limits, num_samples):
    # Generate 1D tensors for each axis
    x_values = torch.linspace(x_lim[0], x_lim[1], num_samples[0])
    y_values = torch.linspace(y_lim[0], y_lim[1], num_samples[1])
    z_values = torch.linspace(z_limits[0], z_limits[1], num_samples[2])

    # Create a 3D grid using torch.meshgrid
    X, Y, Z = torch.meshgrid(x_values, y_values, z_values)

    # Reshape the grid to obtain the sample positions
    samples = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

    return samples

# def get_voxel_centers(x_lim, y_lim, z_lim, x_voxels, y_voxels, z_voxels):
#     # x_step = (x_lim[1] - x_lim[0]) / x_voxels
#     # y_step = (y_lim[1] - y_lim[0]) / y_voxels
#     # z_step = (z_lim[1] - z_lim[0]) / z_voxels

#     # x_centers = torch.linspace(x_lim[0] + x_step / 2, x_lim[1] - x_step / 2, x_voxels)
#     # y_centers = torch.linspace(y_lim[0] + y_step / 2, y_lim[1] - y_step / 2, y_voxels)
#     # z_centers = torch.linspace(z_lim[0] + z_step / 2, z_lim[1] - z_step / 2, z_voxels)
#     x_values = torch.linspace(x_lim[0], x_lim[1], num_samples_x)
#     y_values = torch.linspace(y_lim[0], y_lim[1], num_samples_y)
#     z_values = torch.linspace(z_limits[0], z_limits[1], num_samples_z)

#     # Create a grid of positions using Cartesian product
#     positions = torch.cartesian_prod(x_values, y_values, z_values)
#     return torch.meshgrid(x_centers, y_centers, z_centers)

def get_voxel_centers(x_limits, y_limits, z_limits, num_samples_x, num_samples_y, num_samples_z):
    # Generate evenly spaced values along each axis
    x_values = torch.linspace(x_limits[0], x_limits[1], num_samples_x)
    y_values = torch.linspace(y_limits[0], y_limits[1], num_samples_y)
    z_values = torch.linspace(z_limits[0], z_limits[1], num_samples_z)

    # Create a grid of positions using Cartesian product
    positions = torch.cartesian_prod(x_values, y_values, z_values)

    return positions