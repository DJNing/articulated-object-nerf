# %%
import torch
from models.vanilla_nerf.helper import sample_pdf, sample_along_rays, volumetric_rendering
import torch.nn.functional as F
from datasets import dataset_dict
from torch.utils.data import DataLoader

# %%

def gather_seg_input(seg_feat, part_indicator, samples=None, view_dir=None):
    b, n, _ = seg_feat.shape
    part_ind = part_indicator.repeat([b, n, 1])
    # adding samples and viewdirs are not supported

    cat_list = [seg_feat, part_ind]

    return torch.cat(cat_list, dim=-1)

def get_fine_input(t_mids, weights, origins, directions, t_vals, num_fine_samples, randomized):
    t_vals, samples = sample_pdf(
        bins=t_mids,
        weights=weights[..., 1:-1],
        origins=origins,
        directions=directions,
        t_vals=t_vals,
        num_samples=num_fine_samples,
        randomized=randomized,
    )
    return t_vals, samples

def get_coarse_input(rays, num_coarse_samples, near, far, randomized, lindisp):
    t_vals, samples = sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=lindisp,
                )
    return t_vals, samples

def select_train_pix_ind(mask, seg_label=None, batch_size=1024):
    '''
    mask.shape = [batch_size, N]
    '''
    fg_idx = mask.squeeze().nonzero().squeeze()
    fg_select_inds = torch.randint(0, fg_idx.shape[0], (batch_size, ))
    pix_inds = fg_idx[fg_select_inds]
    return pix_inds

def get_rays_torch(directions, c2w, output_view_dirs = False):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate

    # rays_d = directions @ torch.eye(3).to(directions.device) 
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    # rays_d = torch.matmul(directions, c2w[:, :3].T)
    # rays_d = directions
    #rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].clone().expand(rays_d.shape) # (H, W, 3)

    # if output_radii:
    #     rays_d_orig = directions @ c2w[:, :3].T
    #     dx = torch.sqrt(torch.sum((rays_d_orig[:-1, :, :] - rays_d_orig[1:, :, :]) ** 2, dim=-1))
    #     dx = torch.cat([dx, dx[-2:-1, :]], dim=0)
    #     radius = dx[..., None] * 2 / torch.sqrt(torch.tensor(12, dtype=torch.int8))
    #     radius = radius.reshape(-1)
    
    if output_view_dirs:
        viewdirs = rays_d
        viewdirs /= torch.norm(viewdirs.clone().detach(), dim=-1, keepdim=True)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        viewdirs = viewdirs.view(-1, 3)
        return rays_o, viewdirs, rays_d  
    else:
        rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        return rays_o, rays_d
    
def model_forawrd(batch, opts, model, seg_layers, view_deform, near, far, randomized, white_bkgd):

    pix_inds = select_train_pix_ind(batch['mask'])
    seg_results = []
    rgb_c_results = []
    rgb_f_results = []

    for part in range(opts.part_num):

        # get part indicator
        part_indicator = F.one_hot(torch.tensor(part), opts.part_num).reshape([1, -1])
        part_indicator = part_indicator.to(device=batch['img'].device, dtype=batch['img'].dtype)

        # view deformation
        # input: c2w + articulation + part indicator
        # apply rigid body transformation
        c2w = batch['c2w'].to(torch.float32)
        c2w_r = c2w[:3, :3].reshape([1, -1])
        c2w_t = c2w[:3, -1].reshape([1, -1])
        articulation = batch['art_pose'].reshape([1, -1])

        deform_input = torch.cat((c2w_r, c2w_t, articulation, part_indicator), dim=1).to(torch.float32)
        deform_mat = view_deform(deform_input)
        defrom_r = deform_mat[0, :9].reshape([3, 3])
        deform_t = deform_mat[0, 9:].reshape([3, 1])
        deform_matrix = torch.eye(4).to(device=defrom_r.device, dtype=defrom_r.dtype)
        deform_matrix[:3, :3] = defrom_r
        deform_matrix[:3, -1:] = deform_t

        deform_c2w = torch.matmul(deform_matrix, c2w)

        # get_rays after deformation
        rays_o, viewdirs, rays_d = get_rays_torch(batch['directions'], deform_c2w[:3, :], output_view_dirs=True)

        # something blocking the backprop?

        batch['rays_o'] = rays_o[pix_inds]
        batch['rays_d'] = rays_d[pix_inds]
        batch['viewdirs'] = viewdirs[pix_inds]
        # NeRF
        t_vals_coarse, samples_coarse = get_coarse_input(batch, model.num_coarse_samples, near, far, randomized, model.lindisp)
        corase_dict = {
            "t_vals": t_vals_coarse,
            "samples": samples_coarse
        }
        c_result = model(batch, randomized, white_bkgd, near, far, corase_dict, i_level=0)

        c_rgb, c_acc, c_weights, c_depth = volumetric_rendering(
            c_result["rgb"],
            c_result["sigma"],
            c_result["t_vals"],
            c_result["rays_d"],
            white_bkgd
            )

        t_mids = 0.5 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1])
        t_vals_fine, samples_fine = get_fine_input(
            t_mids, 
            c_weights, 
            batch["rays_o"], 
            batch["rays_d"], 
            c_result["t_vals"], 
            model.num_fine_samples, 
            randomized)

        fine_ip_dict = {
            "t_mids": t_mids,
            "t_vals": t_vals_fine,
            "samples": samples_fine
        }
        f_result = model(batch, randomized, white_bkgd, near, far, fine_ip_dict, i_level=1)

        f_rgb, f_acc, f_weights, f_depth = volumetric_rendering(
            f_result["rgb"],
            f_result["sigma"],
            f_result["t_vals"],
            f_result["rays_d"],
            white_bkgd
            )

        # segmentation
        # seg input: seg_feat + part indicator + samples + view dir
        # seg_feat.shape = [N_rays, num_samples, mlp_netwidth], num_samples unknown 
        
        seg_input = gather_seg_input(f_result["seg_feat"], part_indicator)
        
        seg_init = seg_layers[0](seg_input)
        seg_temp = seg_layers[1](seg_init.squeeze(-1))
        seg_results += [seg_temp]
        
        rgb_c_results += [c_rgb]
        rgb_f_results += [f_rgb]

    # mask the rgb value
    seg_pred = torch.cat(seg_results, dim=-1)

    return seg_pred, pix_inds, rgb_c_results, rgb_f_results

def get_dataloader(opt):
    kwargs_train = {
            "root_dir": opt.root_dir,
            "img_wh": tuple(opt.img_wh),
            "white_back": opt.white_back,
            "model_type": "vailla_nerf",
        }

    kwargs_val = {
                "root_dir": opt.root_dir,
                "img_wh": tuple(opt.img_wh),
                "white_back": opt.white_back,
                "model_type": "vanilla_nerf",
            }

    dataset = dataset_dict[opt.dataset_name]
    train_dataset = dataset(split="train", **kwargs_train)
    val_dataset = dataset(split="val", **kwargs_val)
    train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                num_workers=2,
                batch_size=1,
                pin_memory=True,
            )
    val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                num_workers=2,
                batch_size=1,  
                pin_memory=True,
            )
    return train_loader, val_loader

def gather_part_rgb(seg_one_hot, rgb_pred_list, rgb_target):
    pred_list = []
    target_list = []
    for i in range(len(rgb_pred_list)):
        cur_idx = seg_one_hot[:, i:i+1].nonzero().squeeze()
        cur_pred = rgb_pred_list[i][cur_idx]
        cur_target = rgb_target[cur_idx]
        pred_list += [cur_target]
        target_list += [cur_pred]
        
    return torch.cat(pred_list, dim=0), torch.cat(target_list, dim=0)

