import os
from random import random
from typing import *

from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from datasets import dataset_dict
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributed as dist
from collections import defaultdict
from models.utils import store_image, write_stats, get_obj_rgbs_from_segmap
from torchvision.utils import make_grid
import models.vanilla_nerf.helper as helper
from utils.train_helper import *
from models.vanilla_nerf.util import *
from models.interface import LitModel
import wandb
import random
from utils.viewpoint import view2pose_torch, pose2view_torch, change_apply_change_basis_torch, view2pose_torch_batch, pose2view_torch_batch, convert_ori_torch
from utils.rotation import R_from_quaternions
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)
random.seed(0)

class ArticulationEstimation(nn.Module):
    '''
    Current implemetation for revolute only
    '''
    def __init__(self, mode='qua', perfect_init=False) -> None:
        super().__init__()
        if mode == 'qua':
            pass
        elif mode == 'rad': #radian
            pass
        elif mode == 'deg': # degree
            pass
        else:
            raise RuntimeError('mode == %s for ArticulationEstimation is not defined' % mode)
        
        if perfect_init:
            # perfect init
            init_Q = torch.Tensor([0.97237, 0, -0.233445, 0]) # asset.set_qpos(np.inf * asset.dof)
            axis_origin = convert_ori_torch(torch.Tensor([0, -0.007706040424053753, -0.24714714808389615]))
        # normal init
        else:
            init_Q = torch.Tensor([1, 0, 0, 0])
            axis_origin = torch.Tensor([ 0, 0, 0])

        # axis angle can be obtained from quaternion
        # axis_direction = torch.Tensor([0, 0, 0])

        self.Q = nn.Parameter(init_Q, requires_grad = True)
        self.axis_origin = nn.Parameter(axis_origin, requires_grad = True)
        # self.axis_direction = nn.Parameter(axis_direction, requires_grad = True)


    def forward(self, c2w) -> torch.Tensor():
        '''
        input: c2w
        '''
        E1 = view2pose_torch_batch(c2w)
        translation_matrix = torch.eye(4).to(c2w)
        translation_matrix[:3, 3] = self.axis_origin.view([3])
        rotation_matrix = torch.eye(4).to(c2w)
        R = R_from_quaternions(self.Q)
        rotation_matrix[:3, :3] = R
        E2 = change_apply_change_basis_torch(E1, rotation_matrix, translation_matrix)
        view = pose2view_torch_batch(E2)
        return view
    
class NeRFMLPSeg(nn.Module):
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
        num_density_channels: int = 1,
        num_part: int = 2,
        use_res_seg: bool = True,
        use_part_condition: bool = True,
        res_raw: bool = False,
        include_bg: bool = False
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFMLPSeg, self).__init__()

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
        self.netwidth_condition = netwidth_condition 
        
        self.use_res_seg = use_res_seg
        self.res_raw = res_raw
        if self.use_res_seg:
            # self.seg_layer = nn.Linear(netwidth_condition + pos_size, num_part + 1)
            if self.res_raw:
                # use raw position instead of encoded position
                seg_in_dim = netwidth + 3
            else:
                seg_in_dim = netwidth + pos_size
        else:
            # self.seg_layer = nn.Linear(netwidth_condition, num_part+1)
            seg_in_dim = netwidth
        self.use_part_condition = use_part_condition
        if self.use_part_condition:
            seg_in_dim += num_part
            seg_out_dim = 1
        else:
            if include_bg:
                seg_out_dim = num_part + 1
            else:
                seg_out_dim = num_part
        self.seg_out_dim = seg_out_dim
        self.seg_layer = nn.Linear(seg_in_dim, seg_out_dim)
        self.num_part = num_part
        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)
        init.xavier_uniform_(self.seg_layer.weight)

    def forward(self, x, condition, part_code, pos_raw):
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

        if self.use_res_seg:
            if self.res_raw:
                seg_input = torch.cat((x, pos_raw.reshape(-1, 3)), dim=-1)
            else:
                seg_input = torch.cat((x, inputs), dim=-1)
        else:
            seg_input = x

        if self.use_part_condition:
            seg_input = torch.cat((seg_input, part_code), dim=-1)
        raw_seg = self.seg_layer(seg_input).reshape(-1, num_samples, self.seg_out_dim)

        bottleneck = self.bottleneck_layer(x)
        condition_tile = torch.tile(condition[:, None, :], (1, num_samples, 1)).reshape(
            -1, condition.shape[-1]
        )
        x = torch.cat([bottleneck, condition_tile], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)

        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)
        
        
        ret_dict = {
            "raw_rgb": raw_rgb,
            "raw_density": raw_density,
            "raw_seg": raw_seg
        }

        return ret_dict

class SegmentationModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def forward(self, x):
        pass

class NeRFSeg(nn.Module):
    def __init__(
        self,
        hparams: dict,
        num_levels: int = 2,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        use_viewdirs: bool = True,
        noise_std: float = 0.0,
        lindisp: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRFSeg, self).__init__()
        self.hparams = hparams
        self.rgb_activation = nn.Sigmoid() 
        self.sigma_activation = nn.ReLU()
        if hparams.use_part_condition:
            self.seg_activation = self.sigma_activation
        else:
            self.seg_activation = nn.Softmax(dim=-1)
        self.coarse_mlp = NeRFMLPSeg(min_deg_point, max_deg_point, \
                                    deg_view, use_part_condition=hparams.use_part_condition,
                                    res_raw=self.hparams.res_raw, \
                                    include_bg=self.hparams.include_bg)
        self.fine_mlp = NeRFMLPSeg(min_deg_point, max_deg_point, \
                                    deg_view, use_part_condition=hparams.use_part_condition,
                                    res_raw=self.hparams.res_raw, \
                                    include_bg=self.hparams.include_bg)
        if self.hparams.one_hot_activation:
            self.one_hot_activation = OneHotActivation
        else:
            self.one_hot_activation = None
    # def forward_img(self, batch, randomized, white_bkgd, near, far, skip_seg=False):
    #     '''
    #     Used during test time to render the whole image
    #     '''
    #     c2w = batch['c2w'].to(torch.float32)
    #     rays_o, viewdirs, rays_d = get_rays_torch(batch['directions'], c2w[:3, :], output_view_dirs=True)

    #     # split it chunks for training to avoid oom
    #     chunk_len = rays_o.shape[0] // self.hparams.chunk + 1
    #     chunk_idxs = torch.arange(0, chunk_len) * self.hparams.chunk
    #     chunk_idxs[-1] = rays_o.shape[0]
    #     ret_list = []
    #     for i in range(len(chunk_idxs) - 1):
    #         mini_batch = {}
    #         begin, end = chunk_idxs[i], chunk_idxs[i+1]
    #         mini_batch['rays_o'] = rays_o[begin: end]
    #         mini_batch['rays_d'] = rays_d[begin: end]
    #         mini_batch['viewdirs'] = viewdirs[begin: end]
    #         mini_ret = self.forward(mini_batch, randomized, white_bkgd, near, far)

    #         ret_list += [mini_ret]

    #     # combine results

    #     combined_ret = {}

    #     # Iterate through the keys in the nested dictionary (e.g., 'level_0', 'level_1', etc.)
    #     for level_key in ret_list[0].keys():
    #         combined_ret[level_key] = {}
    #         # Iterate through the mini-ret results for each level
    #         for mini_ret in ret_list:
    #             # Concatenate the values for each key from the mini-ret results
    #             combined_ret[level_key].update({k: torch.cat([v[level_key][k] for v in ret_list], dim=0)})

    #     return combined_ret

    def forward_c2w(self, batch, randomized, white_bkgd, near, far):
        '''
        Used during train time, transform each ray with corresponding c2w
        '''
        c2w = batch['c2w'].to(torch.float32)
        rays_o, viewdirs, rays_d = get_rays_torch_multiple_c2w(batch['dirs'], c2w[:, :3, :], output_view_dirs=True)
        rays = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'viewdirs': viewdirs,
            'part_code': batch.get('part_code', None)
        }
        render_dict = self.forward(rays, randomized, white_bkgd, near, far)
        return render_dict

    def forward(self, rays, randomized, white_bkgd, near, far, seg_feat=False):
        ret = {}
        for i_level in range(self.num_levels):
            if i_level == 0:
                t_vals, samples = helper.sample_along_rays(
                    rays_o=rays["rays_o"],
                    rays_d=rays["rays_d"],
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                )
                mlp = self.coarse_mlp
                output_feat = False

            else:
                t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
                t_vals, samples = helper.sample_pdf(
                    bins=t_mids,
                    weights=ret['level_0']['weights'][..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                )
                mlp = self.fine_mlp
                if seg_feat:
                    output_feat = True

            samples_enc = helper.pos_enc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
            )
            viewdirs_enc = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)
            part_code = rays.get('part_code', None)
            
            if part_code is None:
                raise RuntimeError('part_code not found')
            part_num = part_code.shape[-1]
            part_code = part_code.unsqueeze(1).repeat(1, samples_enc.shape[1], 1)
            part_code = part_code.view([-1, part_num])
            if self.hparams.use_part_condition:
                forward_dict = {
                    "x":samples_enc,
                    "condition":viewdirs_enc,
                    "part_code":part_code,
                    "pos_raw":samples
                }
                
            else:
                forward_dict = {
                    "x":samples_enc,
                    "condition":viewdirs_enc,
                    "part_code":None,
                    "pos_raw":samples
                }

            mlp_ret_dict = mlp(**forward_dict)
            raw_rgb = mlp_ret_dict['raw_rgb']
            raw_density = mlp_ret_dict['raw_density']
            
            raw_seg = mlp_ret_dict['raw_seg']

            if self.noise_std > 0 and randomized:
                raw_density = raw_density + torch.rand_like(raw_density) * self.noise_std

            rgb = self.rgb_activation(raw_rgb)
            density = self.sigma_activation(raw_density)
            seg = self.seg_activation(raw_seg)
            if self.one_hot_activation is not None:
                seg = self.one_hot_activation(seg)
            # render_dict = helper.volumetric_rendering_with_seg(
            #     rgb, 
            #     density,
            #     t_vals,
            #     rays["rays_d"],
            #     white_bkgd=white_bkgd,
            #     seg=seg,
            #     mode=self.hparams.seg_mode
            # )
            if self.hparams.use_part_condition:
                render_dict = helper.volumetric_seg_rendering(
                    rgb, 
                    density,
                    t_vals,
                    rays["rays_d"],
                    white_bkgd=white_bkgd,
                    seg=seg
                )
            elif self.hparams.render_seg:
                helper.volumetric_rendering_with_seg(

                )
                pass
            else:
                # select the right seg to feed in rendering
                # get idx from part_code
                idx = torch.argmax(part_code, dim=-1)
                idx = idx.view(seg.shape[0], seg.shape[1], 1)
                seg_mask = torch.gather(seg, -1, idx)
                render_dict = helper.volumetric_rendering_seg_mask(
                    rgb,
                    density,
                    t_vals,
                    rays['rays_d'],
                    white_bkgd=white_bkgd,
                    seg_mask=seg_mask,
                    seg=seg
                )

            # save for sample_pdf function for fine mlp
            # weights = render_dict['weights']

            # ret.append((comp_rgb, acc, depth, seg_result))
            # feat_out = mlp_ret_dict.get("feat", None)
            result = {
                "rgb": render_dict['comp_rgb'],
                "rgb_seg": render_dict['comp_rgb_seg'],
                "acc": render_dict['acc'],
                "weights": render_dict['weights'],
                "depth": render_dict['depth'],
                "comp_seg": render_dict['comp_seg'],
                "density": density,
                "opacity": render_dict['opacity'],
                "sample_seg": seg
            }
            ret['level_' + str(i_level)] = result

        return ret


class LitNeRFSeg_v2(LitModel):
    def __init__(
        self,
        hparams,
        lr_init: float = 1.0e-3,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__", "hparams"]:
                print(name, value)
                setattr(self, name, value)
        self.hparams.update(vars(hparams))
        super(LitNeRFSeg_v2, self).__init__()

        # if self.hparams.inerf:
        #     pass

        self.model = NeRFSeg(self.hparams)
        if self.hparams.nerf_ckpt is not None:
            # ckpt_dict = torch.load(self.hparam.ckpt_path)['state_dict']
            # self.load_state_dict(ckpt_dict, strict=False)
            helper.load_state_dict_and_report(self, self.hparams.nerf_ckpt)
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
            self.train_dataset = dataset(split="train", **kwargs_train)
            self.val_dataset = dataset(split="val", **kwargs_val)
            self.near = self.train_dataset.near
            self.far = self.train_dataset.far
            self.white_bkgd = self.train_dataset.white_back
            # self.criterion = nn.CrossEntropyLoss(reduction='mean')
            self.criterion = CalculateSegLoss(self.hparams.seg_mode)

    def model_forward(self, batch):
        
        seg_results = []
        rgb_c_results = []
        rgb_f_results = []
        # deform_c2ws = []
        part_num = 1
        w, h = self.hparams.img_wh
        ray_batch = self.hparams.ray_batch
        pix_inds = torch.randint(0, h*w, (ray_batch, ))
        
        c2w = batch['c2w'].to(torch.float32)
        rays_o, viewdirs, rays_d = get_rays_torch(batch['directions'], c2w[:3, :], output_view_dirs=True)
        batch['rays_o'] = rays_o[pix_inds]
        batch['rays_d'] = rays_d[pix_inds]
        batch['viewdirs'] = viewdirs[pix_inds]
        ret_dict = self.model.forward(batch, False, self.white_bkgd, self.near, self.far)
        
        return ret_dict

    # def validation_forward(self, batch):
    #     """
    #     split into mini-batch during forward to avoid OOM
    #     """
    #     # generate rays
    #     c2w = batch['c2w'].to(torch.float32)
    #     rays_o, viewdirs, rays_d = get_rays_torch(batch['directions'], c2w[:3, :], output_view_dirs=True)
    #     input_dict = {
    #         "rays_o": rays_o,
    #         "rays_d": rays_d,
    #         "viewdirs": viewdirs
    #     }
        
    #     pass

    def training_step(self, batch, batch_idx):
        # for k, v in batch.items():
        #     batch[k] = v.squeeze(0)
        for k, v in batch.items():
            if k == "obj_idx":
                continue
            batch[k] = v.squeeze(0)

        # get_rays first for debugging
                
        # rays_o, view_dirs, rays_d = get_rays_torch(batch['directions'], batch['c2w'], True)
        # batch['rays_o'] = rays_o
        # batch['rays_d'] = rays_d
        # batch['viewdirs'] = view_dirs
        
        rendered_results = self.model.forward_c2w(
            batch, self.randomized, self.white_bkgd, self.near, self.far
        )
        
        # if not self.hparams.freeze_nerf:
        rgb_coarse = rendered_results['level_0']['rgb']
        rgb_fine = rendered_results['level_1']['rgb']
        target = batch["rgb"]

        loss0 = helper.img2mse(rgb_coarse, target)
        loss1 = helper.img2mse(rgb_fine, target)
        # loss = loss1 + loss0

        psnr0 = helper.mse2psnr(loss0)
        psnr1 = helper.mse2psnr(loss1)
    
        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)
        
        # seg loss
        seg_target = batch['seg_one_hot']
        acc_coarse = rendered_results['level_0']['acc']
        acc_fine = rendered_results['level_1']['acc']
        seg_coarse = rendered_results['level_0']['comp_seg']
        seg_fine = rendered_results['level_1']['comp_seg']
        has_nan = torch.isnan(seg_coarse).any() + torch.isnan(seg_fine).any() + torch.isnan(seg_target).any()

        if has_nan > 0:
            print('nan in the resutls for segmentation..')

        try:
            seg_coarse_dict = {'pred': seg_coarse, 'gt':seg_target, 'acc':acc_coarse}
            seg_fine_dict = {'pred': seg_fine, 'gt':seg_target, 'acc':acc_fine}
            seg_loss_0 = self.criterion(seg_coarse_dict)
            seg_loss_1 = self.criterion(seg_fine_dict)
        except:
            print("exception for debug")
        self.log("train/seg_1", seg_loss_1, on_step=True, prog_bar=True, logger=True)
        self.log("train/seg_0", seg_loss_0, on_step=True, prog_bar=True, logger=True)
        if self.hparams.freeze_nerf:
            loss = seg_loss_0 + seg_loss_1
        else:
            loss = loss + seg_loss_0 + seg_loss_1
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

    def render_rays_forward_test(self, batch, batch_idx):
        # rays_o, viewdirs, rays_d = get_rays_torch(batch["directions"], batch["c2w"], output_view_dirs=True)
        # batch['rays_o'] = rays_o
        # batch['rays_d'] = rays_d
        # batch['viewdirs'] = viewdirs
        rgb = self.model_forward(batch)
        return {'rgb':rgb}

    def render_rays_test(self, batch, batch_idx):
        # debugging for get_rays_torch
        rays_o, viewdirs, rays_d = get_rays_torch(batch["directions"], batch["c2w"], output_view_dirs=True)
        batch['rays_o'] = rays_o
        batch['rays_d'] = rays_d
        batch['viewdirs'] = viewdirs
        # end modifed by jianning
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
            ret["comp_rgb"] += [rendered_results_chunk['level_1']['comp_rgb']]
            ret["seg"] += [rendered_results_chunk['level_1']['comp_rgb']]
            # for k, v in rendered_results_chunk[1].items():
            #     ret[k] += [v]
        target = batch["target"]
        for k, v in ret.items():
            ret[k] = torch.cat(v, 0)

        test_output = {}
        test_output["target"] = batch["target"]
        test_output["instance_mask"] = batch["instance_mask"]
        test_output["rgb"] = ret["comp_rgb"]
        test_output["seg"] = ret["seg"]
        return test_output

    # def on_sanity_check_start(self):
    #     self.sanity_check = True
        
    # def on_sanity_check_end(self):
    #     self.sanity_check = False

    def on_validation_start(self):
        self.random_batch = np.random.randint(1, size=1)[0]


    def split_forward(self, input_dict):
        """
        Perform forward pass on input data in minibatches.

        Args:
            input_dict (dict): A dictionary containing the input data.
                - "rays_o" (torch.Tensor): Ray origins (N, 3).
                - "rays_d" (torch.Tensor): Ray directions (N, 3).
                - "viewdirs" (torch.Tensor): View directions (N, 3).
        Returns:
            dict: A dictionary containing the gathered results.
                - "rgb" (torch.Tensor): Gathered RGB results (M, 3).
                - "comp_seg" (torch.Tensor): Gathered composition segmentation results (M, 1).
        
        Notes:
            - This function splits the input data into minibatches, applies the forward function to each minibatch, and gathers the results for "rgb" and "comp_seg".
            - The minibatches are determined by the chunk size specified in self.hparams.forward_chunk.
        """
        chunk_size = self.hparams.forward_chunk
        N = input_dict["rays_o"].shape[0]

        # Initialize lists to collect results
        rgb_results = []
        comp_seg_results = []
        depth_results = []

        # Split the data into minibatches
        for i in range(0, N, chunk_size):
            # Get a minibatch of data
            start_idx = i
            end_idx = min(i + chunk_size, N)
            minibatch_data = {
                "rays_o": input_dict["rays_o"][start_idx:end_idx],
                "rays_d": input_dict["rays_d"][start_idx:end_idx],
                "viewdirs": input_dict["viewdirs"][start_idx:end_idx],
            }

            # Call the forward function with the minibatch data
            minibatch_result = self.model.forward(minibatch_data, False, self.white_bkgd, self.near, self.far)
            # Append the result to the list
            rgb_results.append(minibatch_result["level_1"]["rgb"])
            comp_seg_results.append(minibatch_result["level_1"]["comp_seg"])
            depth_results.append(minibatch_result["level_1"]["depth"])

        # Concatenate results from all minibatches
        final_rgb = torch.cat(rgb_results, dim=0)
        final_comp_seg = torch.cat(comp_seg_results, dim=0)
        final_depth = torch.cat(depth_results, dim=0)

        # Return the gathered results as a dictionary
        gathered_results = {
            "rgb": final_rgb,
            "comp_seg": final_comp_seg,
            "depth": final_depth
        }

        return gathered_results


    def validation_step(self, batch, batch_idx):
        """
        batch = {
                "img": img,
                "seg": seg,
                "seg_one_hot": seg_one_hot.to(torch.float32),
                "c2w": c2w,
                "mask": valid_mask,
                "w": w,
                "h": h,
                "directions":self.directions
            }
        """
        
        for k, v in batch.items():
            if k == "obj_idx":
                continue
            batch[k] = v.squeeze(0)
        # generate rays

        W, H = self.hparams.img_wh
        c2w = batch['c2w'].to(torch.float32)
        rays_o, viewdirs, rays_d = get_rays_torch(batch['directions'], c2w[:3, :4], output_view_dirs=True)
        test_rays_d = batch['original_rays_d']
        test_viewdirs = batch['original_viewdirs']
        a = ((viewdirs - test_viewdirs) == 0).all()
        b = ((test_rays_d - rays_d) == 0).all()
        input_dict = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'viewdirs': viewdirs
        }
        val_results = self.split_forward(input_dict)
        
        # rgb metric
        rgb_target = batch['img']
        rgb_pred = val_results['rgb']
        rgb_loss = helper.img2mse(rgb_pred, rgb_target)
        psnr = helper.mse2psnr(rgb_loss)
        # seg metric
        seg_target = batch['seg_one_hot']
        seg_pred = val_results['comp_seg']
        # seg_loss = self.criterion(seg_pred, seg_target)
        seg_metric = calculate_classification_metrics(seg_pred, seg_target)
        # self.log("val/psnr", psnr, on_step=True, prog_bar=True, logger=True)
        # self.log("val/seg_loss", seg_loss, on_step=True, prog_bar=True, logger=True)
        val_results['psnr'] = psnr
        val_results['seg_metric'] = seg_metric
        val_results['img'] = batch['img']
        val_results['seg_one_hot'] = seg_target
        # log rgb and pred
        return val_results

    # def on_validation_epoch_end

    def validation_epoch_end(self, outputs):
        seg_metrics = [ret['seg_metric'] for ret in outputs]
        recall = sum([ret['recall'] for ret in seg_metrics]) / len(outputs)
        precision = sum([ret['precision'] for ret in seg_metrics]) / len(outputs)
        accuracy = sum([ret['accuracy'] for ret in seg_metrics]) / len(outputs)
        psnr = sum(ret['psnr'] for ret in outputs) / len(outputs)
        self.log("val/psnr", psnr, on_epoch=True)
        self.log("val/recall", recall, on_epoch=True)
        self.log("val/precision", precision, on_epoch=True)
        self.log("val/acc", accuracy, on_epoch=True)

        # log 5 rgb and gt
        num_vis = min(5, len(outputs))
        img_list = []
        for i in range(num_vis):
            ret_dict = outputs[i]
            vis_batch = {
                'target': ret_dict['img'],
                'seg_one_hot': ret_dict['seg_one_hot']
            }
            vis_res = {
                'comp_rgb': ret_dict['rgb'],
                'comp_seg': ret_dict['comp_seg']
            }
            # vis_img = visualize_val_rgb(self.hparams.img_wh, vis_batch, vis_res)
            vis_img = visualize_val_rgb_seg(self.hparams.img_wh, vis_batch, vis_res)
            img_list += [vis_img]
        if self.sanity_check:
            self.logger.log_image(key="val/sanity_check", images = img_list)
        else:
            self.logger.log_image(key="val/results", images = img_list)
        return 

    def test_step(self, batch, batch_idx):
        W, H = self.hparams.img_wh
        c2w = batch['c2w'].to(torch.float32)
        rays_o, viewdirs, rays_d = get_rays_torch(batch['directions'], c2w[:3, :], output_view_dirs=True)
        input_dict = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'viewdirs': viewdirs
        }
        ret = self.split_forward(input_dict)
        
        ret["target"] = batch["target"]
        ret["instance_mask"] = batch["instance_mask"]
        return ret

    def configure_optimizers(self):
        if self.hparams.freeze_nerf:
            seg_params = []
            for name, param in self.model.named_parameters():
                if 'seg' in name:
                    seg_params += [param]
                else:
                    param.requires_grad = False
            # seg_params = [param for name, param in self.model.named_parameters() if 'seg' in name]
            return torch.optim.Adam(
                params=iter(seg_params), lr=self.lr_init, betas=(0.9, 0.999)
                )
        else:
            return torch.optim.Adam(
                params=self.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
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
            batch_size=self.hparams.batch_size,
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
        seg_result = self.alter_gather_cat(
            outputs, "comp_seg", all_image_sizes
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
            image_dir = os.path.join(self.hparams.output_path, self.hparams.exp_name, 'test_imgs'
            )
            os.makedirs(image_dir, exist_ok=True)
            img_list = store_image(image_dir, rgbs, "image")
            print('images are stored in: ', image_dir)
            result_path = os.path.join("ckpts", self.hparams.exp_name, "results.json")
            write_stats(result_path, psnr, ssim, lpips, psnr_obj)
            self.logger.log_image(key = "test/results", images = img_list)
        return psnr, ssim, lpips
# =========================================================================================
# =================================== end of v2 ===========================================
# =========================================================================================




def get_rays_torch_multiple_c2w(directions, c2w, output_view_dirs=False):
    """
    Get ray origin and normalized directions in world coordinates for all pixels in one image.

    Inputs:
        directions: (N, 3) precomputed ray directions in camera coordinates
        c2w: (N, 3, 4) transformation matrix from camera coordinates to world coordinates
        output_view_dirs: If True, also output view directions.

    Outputs:
        rays_o: (N, 3), the origin of the rays in world coordinates
        rays_d: (N, 3), the normalized direction of the rays in world coordinates
        viewdirs (optional): (N, 3), the view directions in world coordinates
    """
    # Calculate rays_d (directions in world coordinates)
    c2w_T = c2w[:, :, :3].transpose(1, 2) # (N, 3, 3)
    dirs = directions.unsqueeze(1) # (N, 1, 3)
    rays_d = torch.matmul(dirs, c2w_T)  # (N, 1, 3)
    rays_d = rays_d.squeeze(1) # (N, 3)

    # Calculate rays_o (ray origins in world coordinates)
    rays_o = c2w[:, :, 3]  # (N, 3)

    if output_view_dirs:
        # Normalize view directions
        viewdirs = rays_d.clone()
        viewdirs /= torch.norm(viewdirs.clone(), dim=-1, keepdim=True)  # (N, 3)
        return rays_o, viewdirs, rays_d
    else:
        # Normalize rays_d
        rays_d /= torch.norm(rays_d, dim=1, keepdim=True)  # (N, 3)
        return rays_o, rays_d


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
    rays_d = directions @ c2w[:3, :3].T # (H, W, 3)
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)
    
    if output_view_dirs:
        viewdirs = rays_d.clone()
        viewdirs /= torch.norm(viewdirs.clone(), dim=-1, keepdim=True)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        viewdirs = viewdirs.view(-1, 3)
        return rays_o, viewdirs, rays_d  
    else:
        rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)
        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        return rays_o, rays_d

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

def calculate_classification_metrics(predict_logits, target_one_hot):
    # Convert logits to predicted class labels
    predicted_labels = torch.argmax(predict_logits, dim=1)

    # Convert one-hot target to actual class labels
    true_labels = torch.argmax(target_one_hot, dim=1)

    # Calculate metrics
    true_positive = torch.sum((true_labels == predicted_labels) & (true_labels == 1)).item()
    false_positive = torch.sum((true_labels != predicted_labels) & (predicted_labels == 1)).item()
    false_negative = torch.sum((true_labels != predicted_labels) & (predicted_labels == 0)).item()
    true_negative = torch.sum((true_labels == predicted_labels) & (true_labels == 0)).item()

    precision = true_positive / (true_positive + false_positive + 1e-10)
    recall = true_positive / (true_positive + false_negative + 1e-10)
    accuracy = (true_positive + true_negative) / (len(true_labels) + 1e-10)

    # Store results in a dictionary
    metrics_dict = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }

    return metrics_dict

def compute_multiclass_metrics(logits, ground_truth):
    """
    Compute mIoU, TP, FP, TN, FN for multiclass segmentation.

    Args:
        logits (torch.Tensor): Predicted logits with shape [N, k].
        ground_truth (torch.Tensor): Ground truth one-hot tensor with shape [N, k].

    Returns:
        dict: A dictionary containing the computed metrics (mIoU, TP, FP, TN, FN) for each class.
    """
    # Compute predicted labels (class with the highest probability)
    predicted_labels = torch.argmax(logits, dim=1)
    
    # Initialize dictionaries to store metrics for each class
    iou_dict = {}
    tp_dict = {}
    fp_dict = {}
    tn_dict = {}
    fn_dict = {}
    
    for class_idx in range(logits.shape[1]):
        # Get binary masks for the class
        class_mask_pred = (predicted_labels == class_idx)
        class_mask_gt = (ground_truth[:, class_idx] == 1)
        
        # Calculate intersection, union, and TP, FP, TN, FN
        intersection = torch.logical_and(class_mask_pred, class_mask_gt).sum().item()
        union = torch.logical_or(class_mask_pred, class_mask_gt).sum().item()
        tp = intersection
        fp = (class_mask_pred.sum() - intersection)
        tn = ((~class_mask_pred) & (~class_mask_gt)).sum().item()
        fn = ((~class_mask_pred) & class_mask_gt).sum().item()
        
        # Calculate IoU (Jaccard Index)
        iou = intersection / union if union > 0 else 0
        
        # Store metrics in dictionaries
        iou_dict[f"Class_{class_idx}"] = iou
        tp_dict[f"Class_{class_idx}"] = tp
        fp_dict[f"Class_{class_idx}"] = fp
        tn_dict[f"Class_{class_idx}"] = tn
        fn_dict[f"Class_{class_idx}"] = fn
    
    # Calculate mIoU (average IoU across classes)
    mIoU = sum(iou_dict.values()) / len(iou_dict)
    
    # Create a dictionary to store all metrics
    metrics_dict = {
        "mIoU": mIoU,
        "IoU_per_class": iou_dict,
        "TP_per_class": tp_dict,
        "FP_per_class": fp_dict,
        "TN_per_class": tn_dict,
        "FN_per_class": fn_dict,
    }
    
    return metrics_dict



class CalculateSegLoss(nn.Module):
    def __init__(self, mode):
        super(CalculateSegLoss, self).__init__()
        self.mode = mode
        if mode == 'v1':
            self.criterion = nn.BCELoss()
        elif mode == 'v2' or mode == 'v3':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise RuntimeError('mode %d not supported' % mode)
    
    def forward(self, pair):
        pred, gt = pair['pred'], pair['gt']
        if self.mode == 'v1': # softmax on foreground pixels with BCE
            pred = pred[:, 1:]
            gt = gt[:, 1:]

            pred = nn.functional.softmax(pred, dim=-1)
        elif self.mode == 'v3':           
                # use acc as fg/bg indicator
            acc = pair['acc']
            # use gt as fg indicator
            # acc = gt[:, 1:].sum(dim=-1)
            gt = helper.filter_seg_from_acc(gt, acc)
            pred = helper.filter_seg_from_acc(pred, acc)

        return self.criterion(pred, gt)

class LitNeRFSegArt(LitModel):
    def __init__(
        self,
        hparams,
        lr_init: float = 1.0e-3,
        lr_final: float = 5.0e-5,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__", "hparams"]:
                print(name, value)
                setattr(self, name, value)
        self.hparams.update(vars(hparams))
        super(LitNeRFSegArt, self).__init__()
        self.hparams.white_back = False
        # load pre-trained NeRF model
        self.model = NeRFSeg(self.hparams)
        if self.hparams.nerf_ckpt is not None:
            helper.load_state_dict_and_report(self, self.hparams.nerf_ckpt)

        self.part_num = self.hparams.part_num
        self.lr_final = self.hparams.lr_final
        self.art_list = []
        for _ in range(self.part_num - 1):
            self.art_list += [ArticulationEstimation(perfect_init=True)]
        
        if self.hparams.one_hot_loss:
            self.one_hot_loss = OneHotLoss()
        else:
            self.one_hot_loss = None



        pass

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]

        kwargs_train = {
            "root_dir": self.hparams.root_dir,
            "img_wh": tuple(self.hparams.img_wh),
            "white_back": self.hparams.white_back,
            "model_type": "vailla_nerf",
            "record_hard_sample": self.hparams.record_hard_sample,
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
            self.train_dataset = dataset(split="train", **kwargs_train)
            self.val_dataset = dataset(split="val", **kwargs_val)
            self.near = self.train_dataset.near
            self.far = self.train_dataset.far
            self.white_bkgd = self.train_dataset.white_back
            # self.criterion = nn.CrossEntropyLoss(reduction='mean')
            self.criterion = CalculateSegLoss(self.hparams.seg_mode)
            self.opacity_criterion = nn.BCELoss()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
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

    def configure_optimizers(self):
        seg_params = []
        for name, param in self.model.named_parameters():
            if 'seg' in name:
                seg_params += [param]

        # art_params = []
        # for art_est in self.art_list:
        #     if art_est is not None:
        #         for _, param in art_est.named_parameters():
        #             param.requires_grad = True
        #             art_params += [param]
        seg_opt_dict = {
            'params': seg_params,
            'lr': self.lr_init
        }

        # art_opt_dict = {
        #     'params': art_params,
        #     'lr': self.lr_init
        # }
        return torch.optim.Adam(
            params=[seg_opt_dict], lr=self.lr_init, betas=(0.9, 0.999)
            )

    def training_step(self, batch, batch_idx):

        for k, v in batch.items():
            if k == "obj_idx":
                continue
            batch[k] = v.squeeze(0)

        # transform c2w
        c2w = batch['c2w'].to(torch.float32)
        ray_num = c2w.shape[0]
        # dirs = batch['dirs'].repeat(self.part_num, 1)
        new_c2w_list = []
        part_code_list = []
        for i in range(self.part_num):
            # one_hot = F.one_hot(torch.Tensor(i+1).long(), self.part_num).reshape([1, -1]).repeat(ray_num, 1).to(c2w)
            if self.hparams.include_bg:
                one_hot = torch.zeros(ray_num, self.part_num + 1)
                one_hot[:, i+1] = 1
            else:
                one_hot = torch.zeros(ray_num, self.part_num)
                one_hot[:, i] = 1
            one_hot = one_hot.to(c2w)
            part_code_list += [one_hot]
            if i == 0:
                new_c2w_list += [c2w]
            else:
                new_c2w = self.art_list[i-1](c2w)
                new_c2w_list += [new_c2w]

        part_code = torch.cat(part_code_list, dim=0)
        # batch['part_code'] = part_code
        # batch['c2w_new'] = torch.cat(new_c2w_list, dim=0)
        # batch['dirs_new'] = batch['dirs'].repeat(self.part_num, 1)

        input_dict = {
            'part_code': part_code,
            'c2w': torch.cat(new_c2w_list, dim=0),
            'dirs': batch['dirs'].repeat(self.part_num, 1)
        }
        # forward
        
        rendered_results = self.model.forward_c2w(
            input_dict, self.randomized, self.white_bkgd, self.near, self.far
        )
        '''
        result = {
                "rgb": render_dict['comp_rgb'],
                "acc": render_dict['acc'],
                "weights": render_dict['weights'],
                "depth": render_dict['depth'],
                "comp_seg": render_dict['comp_seg'],
                "density": density,
                "opacity": render_dict['opacity']
            }
            ret['level_' + str(i_level)] = result
        '''
        rgb_target = batch["rgb"]

        rgb_coarse = rendered_results['level_0']['rgb_seg']
        rgb_fine = rendered_results['level_1']['rgb_seg']

        rgb_coarse = rgb_coarse.view([-1, ray_num, 3]).sum(dim=0)
        rgb_fine = rgb_fine.view([-1, ray_num, 3]).sum(dim=0)

        loss0 = helper.img2mse(rgb_coarse, rgb_target)
        loss1 = helper.img2mse(rgb_fine, rgb_target)
        if self.hparams.record_hard_sample:
            if not self.train_dataset.use_sample_list:
                loss0_dict = self._calculate_loss_and_record_sample(rgb_coarse, rgb_target, batch['idx'])
                loss1_dict = self._calculate_loss_and_record_sample(rgb_fine, rgb_target, batch['idx'])
                loss0 = loss0_dict['loss']
                loss1 = loss1_dict['loss']
                sample_0 = loss0_dict['hard_samples']
                sample_1 = loss1_dict['hard_samples']
                samples = torch.cat((sample_0, sample_1)).unique()
                self.train_dataset.sample_list += [samples]
        else:
            loss0 = helper.img2mse(rgb_coarse, rgb_target)
            loss1 = helper.img2mse(rgb_fine, rgb_target)

        psnr0 = helper.mse2psnr(loss0)
        psnr1 = helper.mse2psnr(loss1)
    
        self.log("train/psnr1", psnr1, on_step=True, prog_bar=True, logger=True)
        self.log("train/psnr0", psnr0, on_step=True, prog_bar=True, logger=True)

        # opacity loss
        opa_target = batch['mask']

        opa_coarse = rendered_results['level_0']['opacity'].view([-1, ray_num]).permute(1, 0)
        opa_fine = rendered_results['level_1']['opacity'].view([-1, ray_num]).permute(1, 0)
        # # [ray_num, part_num]
        max_opa_c, _ = torch.max(opa_coarse, dim=0)
        max_opa_f, _ = torch.max(opa_fine, dim=0)
        
        opa_loss_c = (max_opa_c - opa_target)**2
        opa_loss_f = (max_opa_f - opa_target)**2
        opa_loss = 0.5 * (opa_loss_c.mean() + opa_loss_f.mean())
        self.log("train/opa_loss", opa_loss, on_step=True, prog_bar=True, logger=True)
        opts = self.optimizers()
        for pg in opts.param_groups:
            lr = pg['lr']

        self.log('train/lr', lr, on_step=True, prog_bar=True, logger=True)
        # opacity reguralization
        # opa_sum_c = opa_coarse.sum(dim=0) + 1e-10
        # opa_sum_f = opa_fine.sum(dim=0) + 1e-10
        # opa_prob_c = max_opa_c/opa_sum_c
        # opa_prob_f = max_opa_f/opa_sum_f
        # opa_prob_loss_c = opa_prob_c * torch.log(opa_prob_c+1e-10) + (1-opa_prob_c)*torch.log(1-opa_prob_c+1e-10)
        # opa_prob_loss_f = opa_prob_f * torch.log(opa_prob_f+1e-10) + (1-opa_prob_f)*torch.log(1-opa_prob_f+1e-10)

        # reg_loss = -0.5 * (opa_prob_loss_f.mean() + opa_prob_loss_c.mean())
        # self.log("train/opa_reg_loss", reg_loss, on_step=True, prog_bar=True, logger=True)

        # loss = loss0 + loss1 #+ opa_loss + 0.1*reg_loss
        loss = loss0 + loss1 + 0.1*opa_loss
        self.log("train/loss", loss, on_step=True)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if self.hparams.record_hard_sample:
            if self.train_dataset.use_sample_list:
                # reset
                self.train_dataset.sample_list = []
            else:
                temp_idx = torch.cat(self.train_dataset.sample_list)
                self.train_dataset.sample_list = temp_idx.cpu().numpy().astype(int).tolist()
            self.train_dataset.use_sample_list = not self.train_dataset.use_sample_list 
        return super().training_epoch_end(outputs)

    def _calculate_loss_and_record_sample(self, rgb, target, sample_idxs):
        loss = (rgb - target) ** 2
        mean_loss = loss.mean()

        # save samples with loss larger than mean_loss
        loss = loss.mean(dim=-1)
        hard_samples = sample_idxs[loss > mean_loss]
        ret_dict = {
            'loss': mean_loss,
            'hard_samples': hard_samples
        }
        return ret_dict


    def split_forward(self, input_dict):
        """
        Perform forward pass on input data in minibatches.

        Args:
            input_dict (dict): A dictionary containing the input data.
                - "rays_o" (torch.Tensor): Ray origins (N, 3).
                - "rays_d" (torch.Tensor): Ray directions (N, 3).
                - "viewdirs" (torch.Tensor): View directions (N, 3).
        Returns:
            dict: A dictionary containing the gathered results.
                - "rgb" (torch.Tensor): Gathered RGB results (M, 3).
                - "comp_seg" (torch.Tensor): Gathered composition segmentation results (M, 1).
        
        Notes:
            - This function splits the input data into minibatches, applies the forward function to each minibatch, and gathers the results for "rgb" and "comp_seg".
            - The minibatches are determined by the chunk size specified in self.hparams.forward_chunk.
        """
        chunk_size = self.hparams.forward_chunk
        N = input_dict["rays_o"].shape[0]

        # Initialize lists to collect results
        rgb_results = []
        rgb_seg_results = []
        opacity_results = []
        # comp_seg_results = []
        # depth_results = []

        # Split the data into minibatches
        for i in range(0, N, chunk_size):
            # Get a minibatch of data
            start_idx = i
            end_idx = min(i + chunk_size, N)
            minibatch_data = {
                "rays_o": input_dict["rays_o"][start_idx:end_idx],
                "rays_d": input_dict["rays_d"][start_idx:end_idx],
                "viewdirs": input_dict["viewdirs"][start_idx:end_idx],
                "part_code": input_dict["part_code"][start_idx:end_idx]
            }

            # Call the forward function with the minibatch data
            minibatch_result = self.model.forward(minibatch_data, False, self.white_bkgd, self.near, self.far)
            # Append the result to the list
            rgb_results.append(minibatch_result["level_1"]["rgb"])
            rgb_seg_results.append(minibatch_result["level_1"]["rgb_seg"])
            opacity_results.append(minibatch_result['level_1']['opacity'])
            # comp_seg_results.append(minibatch_result["level_1"]["comp_seg"])
            # depth_results.append(minibatch_result["level_1"]["depth"])

        # Concatenate results from all minibatches
        final_rgb = torch.cat(rgb_results, dim=0)
        final_rgb_seg = torch.cat(rgb_seg_results, dim=0)
        final_opa = torch.cat(opacity_results, dim=0)
        # final_comp_seg = torch.cat(comp_seg_results, dim=0)
        # final_depth = torch.cat(depth_results, dim=0)

        # Return the gathered results as a dictionary
        gathered_results = {
            "rgb": final_rgb,
            "rgb_seg": final_rgb_seg,
            "opacity": final_opa
            # "comp_seg": final_comp_seg,
            # "depth": final_depth
        }

        return gathered_results

    def render_img(self, batch):
        c2w = batch['c2w']
        ray_num = c2w.shape[0]
        part_img_list = []
        img_list = []
        opacity_list = []
        for part in range(self.part_num):
            # part_code = F.one_hot(torch.Tensor(part+1).long(), self.part_num).reshape([1, -1]).repeat(ray_num, 1).to(c2w)
            part_code = torch.zeros([ray_num, self.part_num])
            part_code[:, part] = 1
            part_code = part_code.to(c2w)

            if part == 0:
                c2w_part = c2w
            else:
                c2w_part = self.art_list[part-1](c2w)
            
            rays_o, viewdirs, rays_d = get_rays_torch_multiple_c2w(batch['dirs'], c2w_part[:, :3, :], output_view_dirs=True)
            input_dict = {
                'rays_o': rays_o,
                'rays_d': rays_d,
                'viewdirs': viewdirs,
                'part_code': part_code
            }
            render_dict = self.split_forward(input_dict)
            img_list += [render_dict['rgb'].unsqueeze(0)]
            part_img_list += [render_dict['rgb_seg'].unsqueeze(0)] # p * [1, N, 3]
            opacity_list += [render_dict['opacity'].unsqueeze(0)]
            ret_dict = {
                'part_img': part_img_list,
                'img': img_list,
                'opacity': opacity_list
            }
        return ret_dict

    def validation_step(self, batch, batch_idx):
        """
        batch = {
                "img": img,
                "seg": seg,
                "seg_one_hot": seg_one_hot.to(torch.float32),
                "c2w": c2w,
                "mask": valid_mask,
                "w": w,
                "h": h,
                "directions":self.directions
            }
        """
        # if self.sanity_check:
        #     return
        for k, v in batch.items():
            if k == "obj_idx":
                continue
            batch[k] = v.squeeze(0)
        # img_list = self.render_img(batch)
        ret_dict = self.render_img(batch)
        img_list = ret_dict["part_img"]
        opacity = ret_dict["opacity"]
        final_img = torch.cat(img_list, dim=0).sum(dim=0).view([-1, 3])
        rgb_target = batch['img']
        rgb_loss = helper.img2mse(final_img, rgb_target)
        psnr = helper.mse2psnr(rgb_loss)
        img_list += [final_img]
        ret_dict = {
            'img_list':img_list,
            'psnr': psnr,
            'loss': rgb_loss,
            'img': batch['img'],
            'opacity': opacity,
            'valid_mask': batch.get('valid_mask', None)
        }
        return ret_dict

    def on_sanity_check_start(self):
        self.sanity_check = True
        
    def on_sanity_check_end(self):
        self.sanity_check = False

    def validation_epoch_end(self, outputs):
        # pass sanity check
        # if self.sanity_check:
        #     return
        psnr = sum(ret['psnr'] for ret in outputs) / len(outputs)
        self.log("val/psnr", psnr, on_epoch=True)
        
        log_idx = torch.randint(low=0, high=len(outputs), size=(1,))
        log_output = outputs[log_idx[0]]
        img_list = log_output['img_list']
        opacity = log_output['opacity']
        gt = log_output['img']
        mask = log_output['valid_mask']
        if self.sanity_check:
            log_key = "val/sanity_check"
        else:
            log_key = "val/results"

        part_imgs = img_list[:-1]
        final_img = img_list[-1]
        W, H = self.hparams.img_wh
        def get_img(rgb_tensor, H, W, c=3):
            img = rgb_tensor.view(H, W, c).permute(2, 0, 1).cpu()
            return img
        log_part_imgs = [T.ToPILImage()(get_img(part_img, H, W)) for part_img in part_imgs]
        log_opacity = [T.ToPILImage()(get_img(opa, H, W, c=1)) for opa in opacity]
        log_final_img = get_img(final_img, H, W)
        log_gt_img = get_img(gt, H, W)
        stack = torch.stack([log_gt_img, log_final_img])
        grid = make_grid(stack, nrow=2)
        log_grid = T.ToPILImage()(grid)
        part_key = log_key + '/parts'
        final_key = log_key + '/final'
        for idx, part_img in enumerate(log_part_imgs):
            temp_key = part_key + '_' + str(idx)
            self.logger.log_image(key=temp_key, images = [part_img])

        self.logger.log_image(key=final_key, images = [log_grid])
        self.logger.log_image(key=log_key + '/opacity', images = log_opacity)
        
        # mask = mask.view(H, W).permute(2, 0, 1).cpu()
        # log_mask = T.ToPILImage()(mask)
        # self.logger.log_image(key='gt_fg_mask', images=[log_mask])
        # for i in range(len(self.art_list)): 
        #     art = self.art_list[i]
        #     Q = art.Q
        #     origin = art.axis_origin
        #     key = 'part_' + str(i+1)
        #     self.logger.log(key+"_origin", origin)
        #     self.logger.log(key+"_Q", Q)
        return

    def test_step(self, batch, batch_idx):
        """
        batch = {
                "img": img,
                "seg": seg,
                "seg_one_hot": seg_one_hot.to(torch.float32),
                "c2w": c2w,
                "mask": valid_mask,
                "w": w,
                "h": h,
                "directions":self.directions
            }
        """
        for k, v in batch.items():
            if k == "obj_idx":
                continue
            batch[k] = v.squeeze(0)
        img_list = self.render_img(batch)
        final_img = torch.cat(img_list, dim=0).sum(dim=0).view([-1, 3])
        rgb_target = batch['img']
        rgb_loss = helper.img2mse(final_img, rgb_target)
        psnr = helper.mse2psnr(rgb_loss)
        img_list += [final_img]
        ret_dict = {
            'img_list':img_list,
            'psnr': psnr,
            'loss': rgb_loss,
            'img': batch['img']
        }
        return ret_dict

    def test_epoch_end(self, outputs):
        psnr = sum(ret['psnr'] for ret in outputs) / len(outputs)
        self.log("test/psnr", psnr, on_epoch=True)
        save_path = os.path.join(self.hparams.output_path, self.hparams.exp_name, 'test_imgs')
        W, H = self.hparams.img_wh
        for i in range(len(outputs)):

            img_list = outputs[i]['img_list']
            
            final_img = img_list[-1].view(H, W, 3).permute(2, 0, 1).cpu()

            final_pil = T.ToPILImage()(final_img)

            fname = str(i).zfill(3) + '.png'
            save_fname = os.path.join(save_path, fname)
            final_pil.save(save_fname)
        
        return psnr

class OneHotActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = (input > 0.5).to(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

OneHotActivation = OneHotActivation.apply
    
class OneHotLoss(nn.Module):
    def __init__(self, classification_weight=1.0, regularization_weight=0.1):
        super(OneHotLoss, self).__init__()
        self.classification_weight = classification_weight
        self.regularization_weight = regularization_weight

    def forward(self, predictions, labels):
        # Compute the standard cross-entropy loss for classification
        classification_loss = nn.functional.cross_entropy(predictions, labels)

        # Compute the regularization term to encourage one-hot vectors
        one_hot_labels = torch.eye(predictions.shape[1])[labels]  # Convert labels to one-hot
        regularization_loss = torch.mean((predictions - one_hot_labels)**2)

        # Combine both losses with their respective weights
        total_loss = classification_loss * self.classification_weight + regularization_loss * self.regularization_weight

        return total_loss