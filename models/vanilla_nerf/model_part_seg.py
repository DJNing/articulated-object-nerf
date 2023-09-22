from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import dataset_dict
from typing import *
import models.vanilla_nerf.helper as helper
from utils.train_helper import *
from models.vanilla_nerf.util import *
from models.interface import LitModel
from models.utils import store_image, write_stats, get_obj_rgbs_from_segmap

from collections import defaultdict
import os
import wandb


class NeRFMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,
        max_deg_point,
        deg_view,
        netdepth: int = 8,
        netwidth: int = 256,
        netdepth_condition: int = 1,
        netwidth_condition: int = 128,
        skip_layer: int = 4,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFMLP, self).__init__()

        self.net_activation = nn.ReLU()
        pos_size = ((max_deg_point - min_deg_point) * 2 + 1) * input_ch
        view_pos_size = (deg_view * 2 + 1) * input_ch_view

        init_layer = nn.Linear(pos_size, netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [nn.Linear(netwidth + view_pos_size, netwidth_condition)]
        for idx in range(netdepth_condition - 1):
            layer = nn.Linear(netwidth_condition, netwidth_condition)
            init.xavier_uniform_(layer.weight)
            views_linear.append(layer)

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(netwidth, netwidth)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)
        self.net_width = netdepth
        self.netwidth_condition = netwidth_condition
        # self.part_num = part_num
        # if seg_head:
        #     # part indicator, articulation code, density? 
        #     # support for different shared mlp will be implemented later
        #     self.seg_layer = nn.Linear(netwidth_condition + part_num, 1)
        #     init.xavier_uniform_(self.seg_layer)
        # else:
        #     self.seg_layer = False

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)

    def forward(self, batch, skip_seg=False):
        """
        batch: dictionary
        """
        x = batch['x']
        condition = batch['condition']
        num_samples, feat_dim = x.shape[1:]
        x = x.reshape(-1, feat_dim)
        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density = self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )

        bottleneck = self.bottleneck_layer(x)
        condition_tile = torch.tile(condition[:, None, :], (1, num_samples, 1)).reshape(
            -1, condition.shape[-1]
        )
        x = torch.cat([bottleneck, condition_tile], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)

        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)
        
        # part seg
        # part_idx = batch['part_idx']
        # part_indicator = F.one_hot(torch.tensor(part_idx), self.part_num)
        # part_indicator = part_indicator.to(device=x.device, dtype=x.dtype)
        # n = x.shape[0]
        # part_indicator = part_indicator.unsqueeze(0).repeat(n, 1)
        # seg_input = torch.cat((x, part_indicator), dim=1)
        # seg_result = self.seg_layer(seg_input)
        if skip_seg:
            ret_dict = {
            'raw_rgb': raw_rgb,
            'raw_density': raw_density
        }
        else:
            ret_dict = {
                'raw_rgb': raw_rgb,
                'raw_density': raw_density,
                'seg_feat': x.reshape(-1, num_samples, self.netwidth_condition) 
            }
        

        return ret_dict

def get_coarse_input(rays, num_coarse_samples, near, far, randomized, lindisp):
    t_vals, samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=lindisp,
                )
    return t_vals, samples

def get_fine_input(t_mids, weights, origins, directions, t_vals, num_fine_samples, randomized):
    t_vals, samples = helper.sample_pdf(
        bins=t_mids,
        weights=weights[..., 1:-1],
        origins=origins,
        directions=directions,
        t_vals=t_vals,
        num_samples=num_fine_samples,
        randomized=randomized,
    )
    return t_vals, samples

class NeRFSeg(nn.Module):
    def __init__(
        self,
        num_levels: int = 2,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        use_viewdirs: bool = True,
        noise_std: float = 0.0,
        lindisp: bool = False
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFSeg, self).__init__()
        # if use_defrom:
        #     self.deform = DeformationMLP(**deform_args)
        # else:
        #     self.deform = None
        self.rgb_activation = nn.Sigmoid()
        self.sigma_activation = nn.ReLU()
        self.coarse_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)
        self.fine_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)

    def forward(self, rays, randomized, white_bkgd, near, far, input_dict, i_level, skip_seg=False):
        ret = []
        if i_level == 0:
            # t_vals, samples = helper.sample_along_rays(
            #     rays_o=rays["rays_o"],
            #     rays_d=rays["rays_d"],
            #     num_samples=self.num_coarse_samples,
            #     near=near,
            #     far=far,
            #     randomized=randomized,
            #     lindisp=self.lindisp,
            # )
            t_vals, samples = input_dict['t_vals'], input_dict['samples']
            mlp = self.coarse_mlp

        else:
            # t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            # t_vals, samples = helper.sample_pdf(
            #     bins=t_mids,
            #     weights=weights[..., 1:-1],
            #     origins=rays["rays_o"],
            #     directions=rays["rays_d"],
            #     t_vals=t_vals,
            #     num_samples=self.num_fine_samples,
            #     randomized=randomized,
            # )
            t_mids = input_dict['t_mids']
            t_vals, samples = input_dict['t_vals'], input_dict['samples']
            mlp = self.fine_mlp

        # apply deformation here
        samples_enc = helper.pos_enc(
            samples,
            self.min_deg_point,
            self.max_deg_point,
        )
        viewdirs_enc = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)
        mlp_batch = {
            'x': samples_enc,
            'condition': viewdirs_enc,
            # 'part_idx': part_idx
        }
        mlp_ret_dict = mlp(mlp_batch)
        raw_rgb = mlp_ret_dict['raw_rgb']
        raw_sigma = mlp_ret_dict['raw_density']
        # seg_result = mlp_ret_dict.get('seg', None)

        if self.noise_std > 0 and randomized:
            raw_sigma = raw_sigma + torch.rand_like(raw_sigma) * self.noise_std

        rgb = self.rgb_activation(raw_rgb)
        sigma = self.sigma_activation(raw_sigma)

        # comp_rgb, acc, weights, depth = helper.volumetric_rendering(
        #     rgb,
        #     sigma,
        #     t_vals,
        #     rays["rays_d"],
        #     white_bkgd=white_bkgd,
        # )

        # ret.append((comp_rgb, acc, depth, seg_result))
        ret = {
            "rgb": rgb,
            "sigma": sigma,
            "t_vals": t_vals,
            "rays_d": rays["rays_d"],
            "seg_feat": mlp_ret_dict["seg_feat"]
        }
        return ret
    
class DeformationMLP(nn.Module):
    def __init__(self,  layer_num, layer_width, input_dim, output_dim) -> None:
        super(DeformationMLP, self).__init__()
        layers = []

        # Add the input layer
        layers.append(nn.Linear(input_dim, layer_width))
        layers.append(nn.ReLU())

        for _ in range(layer_num - 2):  # -2 because we've added input and output layers already
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(nn.ReLU()) 

        layers.append(nn.Linear(layer_width, output_dim))

        # Define the MLP as a sequence of layers
        self.mlp = nn.Sequential(*layers)

    
    def forward(self, x):
        """
        input: articulation code, original view point
        output: transformation or code after pose endocer
        """
        return self.mlp(x)
    
class LitNeRFSeg(LitModel):
    def __init__(
                self,
                hparams,
                lr_init: float = 1e-3,
                lr_final: float = 5.0e-6,
                lr_delay_steps: int = 2500,
                lr_delay_mult: float = 0.01,
                randomized: bool = True):
        for name, value in vars().items():
            if name not in ["self", "__class__", "hparams"]:
                print(name, value)
                setattr(self, name, value)
        self.hparams.update(vars(hparams))
        super(LitNeRFSeg, self).__init__()

        self.model = NeRFSeg()
        # self.model.eval() #don't update the original NeRF model. Failed to compute gradient if it's set to eval
        # for _, param in self.model.named_parameters():
        #     param.requires_grad = False


        ckpt_dict = torch.load(hparams.ckpt_path)
        model_state_dict = self.state_dict()
        missing_keys = [key for key in model_state_dict.keys() if key not in ckpt_dict['state_dict'].keys()]
        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys}")
        self.load_state_dict(ckpt_dict['state_dict'], strict=False)
        
        self.init_seg_layer = nn.Linear(128 + self.hparams.part_num, 1)
        comp_dim = self.model.num_coarse_samples + self.model.num_fine_samples + 1
        self.comp_seg_layer = nn.Linear(comp_dim, 1)
        self.criterion = nn.CrossEntropyLoss()
        self.view_deform = DeformationMLP(hparams.deform_layer_num, hparams.deform_layer_width, hparams.deform_input_dim, hparams.deform_output_dim)

        # self.automatic_optimization = False
        self.optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr_init, betas=(0.9, 0.999)
        )
        self.deform_id_loss = torch.nn.MSELoss()
        pass
    
    def setup(self, stage: Optional[str] = None) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]

        kwargs_train = {
            "root_dir": self.hparams.root_dir,
            "img_wh": tuple(self.hparams.img_wh),
            "white_back": self.hparams.white_back,
            "model_type": "vailla_nerf",
        }
        kwargs_val = {
            "root_dir": self.hparams.root_dir,
            "img_wh": tuple(self.hparams.img_wh),
            "white_back": self.hparams.white_back,
            "model_type": "vanilla_nerf",
        }

        if self.hparams.run_eval:
            kwargs_test = {
                "root_dir": self.hparams.root_dir,
                "img_wh": tuple(self.hparams.img_wh),
                "white_back": self.hparams.white_back,
                "model_type": "vanilla_nerf",
                "eval_inference": self.hparams.render_name,
            }
            self.test_dataset = dataset(split="test_val", **kwargs_test)
            self.near = self.test_dataset.near
            self.far = self.test_dataset.far
            self.white_bkgd = self.test_dataset.white_back

        else:
            kwargs_test = {
                "root_dir": self.hparams.root_dir,
                "img_wh": tuple(self.hparams.img_wh),
                "white_back": self.hparams.white_back,
                "model_type": "vanilla_nerf",
                "eval_inference": self.hparams.render_name,
            }
            self.test_dataset = dataset(split="val", **kwargs_test)
            self.train_dataset = dataset(split="train", **kwargs_train)
            self.val_dataset = dataset(split="val", **kwargs_val)
            self.near = self.train_dataset.near
            self.far = self.train_dataset.far
            self.white_bkgd = self.train_dataset.white_back

    def model_forward_test(self, batch, debug=True):
        seg_results = []
        rgb_c_results = []
        rgb_f_results = []
        deform_c2ws = []
        if debug:
            part_num = 1
            w, h = self.hparams.img_wh
            pix_inds = torch.arange(0, w*h)
        else:
            part_num = self.hparams.part_num
            pix_inds = self.select_train_pix_ind(batch['mask'])
        for part in range(part_num):

            # get part indicator
            part_indicator = F.one_hot(torch.tensor(part), self.hparams.part_num).reshape([1, -1])
            part_indicator = part_indicator.to(device=batch['img'].device, dtype=batch['img'].dtype)

            # view deformation
            # input: c2w + articulation + part indicator
            # apply rigid body transformation
            
            c2w = batch['c2w'].to(torch.float32)
            if part == 0:
                deform_c2w = c2w
                
            else:
                c2w_r = c2w[:3, :3].reshape([1, -1])
                c2w_t = c2w[:3, -1].reshape([1, -1])
                articulation = batch['art_pose'].reshape([1, -1])

                deform_input = torch.cat((c2w_r, c2w_t, articulation, part_indicator), dim=1).to(torch.float32)
                deform_mat = self.view_deform(deform_input)
                defrom_r = deform_mat[0, :9].reshape([3, 3])
                deform_t = deform_mat[0, 9:].reshape([3, 1])
                deform_matrix = torch.eye(4).to(device=defrom_r.device, dtype=defrom_r.dtype)
                deform_matrix[:3, :3] = defrom_r
                deform_matrix[:3, -1:] = deform_t

                deform_c2w = torch.matmul(deform_matrix, c2w)

            deform_c2ws += [deform_c2w]

            # get_rays after deformation
            rays_o, viewdirs, rays_d = get_rays_torch(batch['directions'], deform_c2w[:3, :], output_view_dirs=True)
            # something blocking the backprop?

            # get_ray_batch
            # rays, rays_d, view_dirs, src_img, rgbs, mask = self.get_ray_batch(
            #     cam_rays, cam_view_dirs, cam_rays_d, img, seg, ray_batch_size=4096
            # )
            # split into several chunk
            batch['rays_o'] = rays_o[pix_inds]
            batch['rays_d'] = rays_d[pix_inds]
            batch['viewdirs'] = viewdirs[pix_inds]
        
            chunk_len = rays_o.shape[0] // self.hparams.chunk + 1
            chunk_idxs = torch.arange(0, chunk_len) * self.hparams.chunk
            chunk_idxs[-1] = rays_o.shape[0]
            
            frgb_results = []
            for i in range(len(chunk_idxs) - 1):
                batch_chunk = dict()
                for k, v in batch.items():
                    if k == "rays_o" or k == "rays_d" or k == "viewdirs":
                        batch_chunk[k] = v[chunk_idxs[i] : chunk_idxs[i+1]]
                # NeRF
                t_vals_coarse, samples_coarse = get_coarse_input(batch_chunk, self.model.num_coarse_samples, self.near, self.far, self.randomized, self.model.lindisp)
                corase_dict = {
                    "t_vals": t_vals_coarse,
                    "samples": samples_coarse
                }
                c_result = self.model(batch_chunk, self.randomized, self.white_bkgd, self.near, self.far, corase_dict, i_level=0)

                c_rgb, c_acc, c_weights, c_depth = helper.volumetric_rendering(
                    c_result["rgb"],
                    c_result["sigma"],
                    c_result["t_vals"],
                    c_result["rays_d"],
                    self.white_bkgd
                    )

                t_mids = 0.5 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1])
                t_vals_fine, samples_fine = get_fine_input(
                    t_mids, 
                    c_weights, 
                    batch_chunk["rays_o"], 
                    batch_chunk["rays_d"], 
                    c_result["t_vals"], 
                    self.model.num_fine_samples, 
                    self.randomized)

                fine_ip_dict = {
                    "t_mids": t_mids,
                    "t_vals": t_vals_fine,
                    "samples": samples_fine
                }
                f_result = self.model(batch_chunk, self.randomized, self.white_bkgd, self.near, self.far, fine_ip_dict, i_level=1)

                f_rgb, f_acc, f_weights, f_depth = helper.volumetric_rendering(
                    f_result["rgb"],
                    f_result["sigma"],
                    f_result["t_vals"],
                    f_result["rays_d"],
                    self.white_bkgd
                    )
                frgb_results += [f_rgb]
            

            # segmentation
            # seg input: seg_feat + part indicator + samples + view dir
            # seg_feat.shape = [N_rays, num_samples, mlp_netwidth], num_samples unknown 
            if debug:
                pass
            else:

                seg_input = self.gather_seg_input(f_result["seg_feat"], part_indicator)

                seg_init = self.init_seg_layer(seg_input)
                seg_temp = self.comp_seg_layer(seg_init.squeeze(-1))
                seg_results += [seg_temp]
                
                rgb_c_results += [c_rgb]
                rgb_f_results += [f_rgb]
            

            # rgb_coarse = c_result["rgb"]
            # rgb_fine = f_result["rgb"]
            # target = batch["target"]

        if debug:
            seg_pred = None
            return torch.cat(frgb_results, dim=0)
        else:
            # mask the rgb value
            seg_pred = torch.cat(seg_results, dim=-1)

            return seg_pred, pix_inds, rgb_c_results, rgb_f_results, deform_c2ws

    def model_forawrd(self, batch, debug=False):

        
        seg_results = []
        rgb_c_results = []
        rgb_f_results = []
        deform_c2ws = []
        if debug:
            part_num = 1
            w, h = self.hparams.img_wh
            pix_inds = torch.arange(0, w*h)
        else:
            part_num = self.hparams.part_num
            pix_inds = self.select_train_pix_ind(batch['mask'])
        for part in range(part_num):

            # get part indicator
            part_indicator = F.one_hot(torch.tensor(part), self.hparams.part_num).reshape([1, -1])
            part_indicator = part_indicator.to(device=batch['img'].device, dtype=batch['img'].dtype)
            
            c2w = batch['c2w'].to(torch.float32)
            # view deformation
            # input: c2w + articulation + part indicator
            # apply rigid body transformation
            if part != 0:
                c2w_r = c2w[:3, :3].reshape([1, -1])
                c2w_t = c2w[:3, -1].reshape([1, -1])
                articulation = batch['art_pose'].reshape([1, -1])

                deform_input = torch.cat((c2w_r, c2w_t, articulation, part_indicator), dim=1).to(torch.float32)
                deform_mat = self.view_deform(deform_input)
                defrom_r = deform_mat[0, :9].reshape([3, 3])
                deform_t = deform_mat[0, 9:].reshape([3, 1])
                deform_matrix = torch.eye(4).to(device=defrom_r.device, dtype=defrom_r.dtype)
                deform_matrix[:3, :3] = defrom_r
                deform_matrix[:3, -1:] = deform_t

                deform_c2w = torch.matmul(deform_matrix, c2w)
            else:
                deform_c2w = c2w

            deform_c2ws += [deform_c2w]

            # get_rays after deformation
            rays_o, viewdirs, rays_d = get_rays_torch(batch['directions'], deform_c2w[:3, :], output_view_dirs=True)
            # something blocking the backprop?

            # get_ray_batch
            # rays, rays_d, view_dirs, src_img, rgbs, mask = self.get_ray_batch(
            #     cam_rays, cam_view_dirs, cam_rays_d, img, seg, ray_batch_size=4096
            # )

            batch['rays_o'] = rays_o[pix_inds]
            batch['rays_d'] = rays_d[pix_inds]
            batch['viewdirs'] = viewdirs[pix_inds]
            # NeRF
            t_vals_coarse, samples_coarse = get_coarse_input(batch, self.model.num_coarse_samples, self.near, self.far, self.randomized, self.model.lindisp)
            corase_dict = {
                "t_vals": t_vals_coarse,
                "samples": samples_coarse
            }
            c_result = self.model(batch, self.randomized, self.white_bkgd, self.near, self.far, corase_dict, i_level=0)

            c_rgb, c_acc, c_weights, c_depth = helper.volumetric_rendering(
                c_result["rgb"],
                c_result["sigma"],
                c_result["t_vals"],
                c_result["rays_d"],
                self.white_bkgd
                )

            t_mids = 0.5 * (t_vals_coarse[..., 1:] + t_vals_coarse[..., :-1])
            t_vals_fine, samples_fine = get_fine_input(
                t_mids, 
                c_weights, 
                batch["rays_o"], 
                batch["rays_d"], 
                c_result["t_vals"], 
                self.model.num_fine_samples, 
                self.randomized)

            fine_ip_dict = {
                "t_mids": t_mids,
                "t_vals": t_vals_fine,
                "samples": samples_fine
            }
            f_result = self.model(batch, self.randomized, self.white_bkgd, self.near, self.far, fine_ip_dict, i_level=1)

            f_rgb, f_acc, f_weights, f_depth = helper.volumetric_rendering(
                f_result["rgb"],
                f_result["sigma"],
                f_result["t_vals"],
                f_result["rays_d"],
                self.white_bkgd
                )


            # segmentation
            # seg input: seg_feat + part indicator + samples + view dir
            # seg_feat.shape = [N_rays, num_samples, mlp_netwidth], num_samples unknown 
            if debug:
                pass
            else:

                seg_input = self.gather_seg_input(f_result["seg_feat"], part_indicator)

                seg_init = self.init_seg_layer(seg_input)
                seg_temp = self.comp_seg_layer(seg_init.squeeze(-1))
                seg_results += [seg_temp]
            

            # rgb_coarse = c_result["rgb"]
            # rgb_fine = f_result["rgb"]
            rgb_c_results += [c_rgb]
            rgb_f_results += [f_rgb]
            # target = batch["target"]

        if debug:
            seg_pred = None
        else:
            # mask the rgb value
            seg_pred = torch.cat(seg_results, dim=-1)

        return seg_pred, pix_inds, rgb_c_results, rgb_f_results, deform_c2ws
            
    def training_step(self, batch, batch_idx):
        # for k, v in batch.items():
        #     batch[k] = v.squeeze(0)
        for k, v in batch.items():
            if k == "obj_idx":
                continue
            batch[k] = v.squeeze(0) # squeeze the batch_size dim

        seg_pred, pix_inds, c_rgbs, f_rgbs, deform_c2ws = self.model_forawrd(batch)
        supervised = True
        if supervised:
            seg_final = F.softmax(seg_pred, dim=-1)
            seg_target = batch['seg_one_hot'][pix_inds][:, 1:] # remove background label
            seg_loss = self.criterion(seg_final, seg_target)
            
            rgb_target = batch['img'][pix_inds]
            # gather part rgb values
            
            
            
            c_pred, c_target = self.gather_part_rgb(seg_target, c_rgbs, rgb_target)
            f_pred, f_target = self.gather_part_rgb(seg_target, f_rgbs, rgb_target)

            loss0 = helper.img2mse(c_pred, c_target)
            loss1 = helper.img2mse(f_pred, f_target)

            psnr0 = helper.mse2psnr(loss0)
            psnr1 = helper.mse2psnr(loss1)

            # c2w_target = torch.eye(4).to(device=c_pred.device, dtype=c_pred.dtype)

            # deform_loss = self.deform_id_loss(deform_c2ws[0], c2w_target)

            loss = loss0 + loss1 + seg_loss #+ deform_loss
            # self.log("train/deform_loss", deform_loss, on_step=True, prog_bar=True, logger=True)
            self.log("train/seg_loss", seg_loss, on_step=True, prog_bar=True, logger=True)

            # part0_loss = c_loss['part_0'].item() + f_loss['part_0'].item()
            # part1_loss = c_loss['part_1'].item() + f_loss['part_1'].item()
            # part0_psnr = helper.mse2psnr(part0_loss)
            # part1_psnr = helper.mse2psnr(part1_loss)
            self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
            self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
            self.log("train/loss", loss, on_step=True)

        else:
            
            rgb_target = batch['img'][pix_inds]
            seg_target = batch['seg'][pix_inds]
            loss0 = helper.img2mse(c_rgbs, rgb_target)
            loss1 = helper.img2mse(f_rgbs, rgb_target)
            loss = loss1 + loss0

            psnr0 = helper.mse2psnr(loss0)
            psnr1 = helper.mse2psnr(loss1)

            self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
            self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
            self.log("train/loss", loss, on_step=True)
        return loss

    # def render_rays(self, batch, batch_idx):
    #     ret = {}
    #     rendered_results = self.model(
    #         batch, False, self.white_bkgd, self.near, self.far
    #     )
    #     rgb_fine = rendered_results[1][0]
    #     target = batch["target"]
    #     ret["target"] = target
    #     ret["rgb"] = rgb_fine
    #     return ret

    def render_rays(self, batch, batch_idx):
        B = batch["rays_o"].shape[0]
        ret = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            batch_chunk = dict()
            for k, v in batch.items():
                if k == "obj_idx":
                    batch_chunk[k] = v
                if k == "radii":
                    batch_chunk[k] = v[:, i : i + self.hparams.chunk]
                else:
                    batch_chunk[k] = v[i : i + self.hparams.chunk]
            rendered_results_chunk = self.model(
                batch_chunk, False, self.white_bkgd, self.near, self.far
            )
            # here 1 denotes fine
            ret["comp_rgb"] += [rendered_results_chunk[1][0]]
            ret["acc"] += [rendered_results_chunk[1][1]]
            ret["depth"] += [rendered_results_chunk[1][2]]
            # for k, v in rendered_results_chunk[1].items():
            #     ret[k] += [v]
        target = batch["target"]
        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)
        psnr_ = self.psnr_legacy(ret["comp_rgb"], batch["target"]).mean()
        self.log("val/psnr", psnr_.item(), on_step=True, prog_bar=True, logger=True)
        return ret

    def render_rays_test(self, batch, batch_idx):
        B = batch["rays_o"].shape[0]
        ret = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            batch_chunk = dict()
            for k, v in batch.items():
                if k == "radii":
                    batch_chunk[k] = v[:, i : i + self.hparams.chunk]
                else:
                    batch_chunk[k] = v[i : i + self.hparams.chunk]
            rendered_results_chunk = self.model(
                batch_chunk, False, self.white_bkgd, self.near, self.far
            )
            # here 1 denotes fine
            ret["comp_rgb"] += [rendered_results_chunk[1][0]]
            # for k, v in rendered_results_chunk[1].items():
            #     ret[k] += [v]
        target = batch["target"]
        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)

        test_output = {}
        test_output["target"] = batch["target"]
        test_output["instance_mask"] = batch["instance_mask"]
        test_output["rgb"] = ret["comp_rgb"]
        return test_output

    def on_validation_start(self):
        self.random_batch = np.random.randint(1, size=1)[0]

    def validation_step(self, batch, batch_idx):
        for k, v in batch.items():
            print(k, v.shape)

        for k, v in batch.items():
            if k == "obj_idx":
                continue
            batch[k] = v.squeeze(0)
            if k == "radii":
                batch[k] = v.unsqueeze(-1)
            if k == "near_obj" or k == "far_obj":
                batch[k] = batch[k].unsqueeze(-1)

        W, H = self.hparams.img_wh
        # ret = self.render_rays(batch, batch_idx)
        # # rank = dist.get_rank()
        # rank = 0
        # if rank == 0:
        #     if batch_idx == self.random_batch:
        #         grid_img = visualize_val_rgb_opa_depth((W, H), batch, ret)
        #         self.logger.experiment.log({"val/GT_pred rgb": wandb.Image(grid_img)})

        # return self.render_rays(batch, batch_idx)

        return None

    @staticmethod
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
    
    @staticmethod
    def gather_seg_input(seg_feat, part_indicator, samples=None, view_dir=None):
        b, n, _ = seg_feat.shape
        part_ind = part_indicator.repeat([b, n, 1])
        # adding samples and viewdirs are not supported

        cat_list = [seg_feat, part_ind]

        return torch.cat(cat_list, dim=-1)

    def test_step(self, batch, batch_idx):
        for k, v in batch.items():
            if k == "obj_idx":
                continue
            batch[k] = v.squeeze(0) # squeeze the batch_size dim
        # seperate into multiple chunk

        rgb = self.model_forward_test(batch, debug=True)
        w, h = self.hparams.img_wh
        img = rgb.view([h, w, 3])
        image_dir = os.path.join(
            "ckpts", self.hparams.exp_name, 'test'
        )
        os.makedirs(image_dir, exist_ok=True)
        store_image(image_dir, [img], "image")
        return 

    def configure_optimizers(self):
        # params_group = []
        opt_params = []
        for p in self.init_seg_layer.parameters():
            opt_params += [p]

        for p in self.comp_seg_layer.parameters():
            opt_params += [p]

        for p in self.view_deform.parameters():
            opt_params += [p]

        return torch.optim.Adam(
            params=iter(opt_params), lr=self.lr_init, betas=(0.9, 0.999)
        )

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu,
        using_native_amp,
        using_lbfgs,
    ):
        step = self.trainer.global_step
        max_steps = self.hparams.run_max_steps

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=2,
            batch_size=1,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=2,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,
            pin_memory=True,
        )

    # def validation_epoch_end(self, outputs):
    #     val_image_sizes = self.trainer.datamodule.val_image_sizes
    #     rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
    #     targets = self.alter_gather_cat(outputs, "target", val_image_sizes)
    #     psnr_mean = self.psnr_each(rgbs, targets).mean()
    #     ssim_mean = self.ssim_each(rgbs, targets).mean()
    #     lpips_mean = self.lpips_each(rgbs, targets).mean()
    #     self.log("val/psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
    #     self.log("val/ssim", ssim_mean.item(), on_epoch=True, sync_dist=True)
    #     self.log("val/lpips", lpips_mean.item(), on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        # dmodule = self.trainer.datamodule
        # all_image_sizes = (
        #     dmodule.all_image_sizes
        #     if not dmodule.eval_test_only
        #     else dmodule.test_image_sizes
        # )
        all_image_sizes = self.test_dataset.image_sizes

        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        instance_masks = self.alter_gather_cat(
            outputs, "instance_mask", all_image_sizes
        )
        targets = self.alter_gather_cat(outputs, "target", all_image_sizes)

        psnr = self.psnr(rgbs, targets, None, None, None)
        ssim = self.ssim(rgbs, targets, None, None, None)
        lpips = self.lpips(rgbs, targets, None, None, None)

        all_obj_rgbs, all_target_rgbs = get_obj_rgbs_from_segmap(
            instance_masks, rgbs, targets
        )

        psnr_obj = self.psnr(all_obj_rgbs, all_target_rgbs, None, None, None)
        print("psnr obj", psnr_obj)

        # psnr = self.psnr(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        # ssim = self.ssim(rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        # lpips = self.lpips(
        #     rgbs, targets, dmodule.i_train, dmodule.i_val, dmodule.i_test
        # )

        self.log("test/psnr", psnr["test"], on_epoch=True)
        self.log("test/ssim", ssim["test"], on_epoch=True)
        self.log("test/lpips", lpips["test"], on_epoch=True)
        print("psnr, ssim, lpips", psnr, ssim, lpips)
        self.log("test/psnr_obj", psnr_obj["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(
                "ckpts", self.hparams.exp_name, self.hparams.render_name
            )
            os.makedirs(image_dir, exist_ok=True)
            store_image(image_dir, rgbs, "image")

            result_path = os.path.join("ckpts", self.hparams.exp_name, "results.json")
            write_stats(result_path, psnr, ssim, lpips, psnr_obj)

        return psnr, ssim, lpips

    def select_train_pix_ind(self, mask, seg_label=None, batch_size=1024):
        '''
        mask.shape = [batch_size, N]
        '''
        w, h = self.hparams.img_wh
        fg_idx = mask.squeeze().nonzero().squeeze()
        fg_select_inds = torch.randint(0, fg_idx.shape[0], (batch_size, ))
        pix_inds = fg_idx[fg_select_inds]
        return pix_inds

    # def get_ray_batch(
    #     self, cam_rays, cam_view_dirs, cam_rays_d, img, part_label, ray_batch_size
    # ):
    #     """
    #     """
    #     # instance_mask = T.ToTensor()(instance_mask)
    #     # img = Image.fromarray(np.uint8(img))
    #     # img = T.ToTensor()(img)

    #     # cam_rays = torch.FloatTensor(cam_rays)
    #     # cam_view_dirs = torch.FloatTensor(cam_view_dirs)
    #     # cam_rays_d = torch.FloatTensor(cam_rays_d)
    #     rays = cam_rays.view(-1, cam_rays.shape[-1])
    #     rays_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
    #     view_dirs = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])

    #     if self.split == "train":
    #         _, H, W = img.shape
    #         pix_inds = torch.randint(0, H * W, (ray_batch_size,))
    #         src_img = self.img_transform(img)
    #         msk_gt = part_label.permute(1, 2, 0).flatten(0, 1)[pix_inds, ...]
    #         rgbs = img.permute(1, 2, 0).flatten(0, 1)[pix_inds, ...]
    #         rays = rays[pix_inds]
    #         rays_d = rays_d[pix_inds]
    #         view_dirs = view_dirs[pix_inds]

    #     else:
    #         src_img = self.img_transform(img)
    #         msk_gt = part_label.permute(1, 2, 0).flatten(0, 1)
    #         rgbs = img.permute(1, 2, 0).flatten(0, 1)

    #     return rays, rays_d, view_dirs, src_img, rgbs, msk_gt

    
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
    # rays_d = directions.clone() @ c2w[:, :3].T # (H, W, 3)
    rays_d = torch.matmul(directions, c2w[:, :3].T)
    #rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

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

def get_ray_batch(
        self, cam_rays, cam_view_dirs, cam_rays_d, img, instance_mask, ray_batch_size
    ):
    instance_mask = T.ToTensor()(instance_mask)
    img = Image.fromarray(np.uint8(img))
    img = T.ToTensor()(img)

    cam_rays = torch.FloatTensor(cam_rays)
    cam_view_dirs = torch.FloatTensor(cam_view_dirs)
    cam_rays_d = torch.FloatTensor(cam_rays_d)
    rays = cam_rays.view(-1, cam_rays.shape[-1])
    rays_d = cam_rays_d.view(-1, cam_rays_d.shape[-1])
    view_dirs = cam_view_dirs.view(-1, cam_view_dirs.shape[-1])

    if self.split == "train":
        _, H, W = img.shape
        pix_inds = torch.randint(0, H * W, (ray_batch_size,))
        src_img = self.img_transform(img)
        msk_gt = instance_mask.permute(1, 2, 0).flatten(0, 1)[pix_inds, ...]
        rgbs = img.permute(1, 2, 0).flatten(0, 1)[pix_inds, ...]
        rays = rays[pix_inds]
        rays_d = rays_d[pix_inds]
        view_dirs = view_dirs[pix_inds]

    else:
        src_img = self.img_transform(img)
        msk_gt = instance_mask.permute(1, 2, 0).flatten(0, 1)
        rgbs = img.permute(1, 2, 0).flatten(0, 1)

    return rays, rays_d, view_dirs, src_img, rgbs, msk_gt


if __name__ == "__main__":
    
    pass