# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
from random import random
from typing import *
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

import models.vanilla_nerf.helper as helper
from utils.train_helper import *
from models.vanilla_nerf.util import *
from models.interface import LitModel
import wandb
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)
random.seed(0)


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
        num_density_channels: int = 1,
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

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)

    def forward(self, x, condition):
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

        return raw_rgb, raw_density


class NeRF(nn.Module):
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
        lindisp: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(NeRF, self).__init__()

        self.rgb_activation = nn.Sigmoid()
        self.sigma_activation = nn.ReLU()
        self.coarse_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)
        self.fine_mlp = NeRFMLP(min_deg_point, max_deg_point, deg_view)

    def  model_forward(self, rays, randomized, white_bkgd, near, far, input_dict, i_level, skip_seg=False):
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
        # mlp_ret_dict = mlp(samples_enc, viewdirs_enc)
        # raw_rgb = mlp_ret_dict['raw_rgb']
        # raw_sigma = mlp_ret_dict['raw_density']
        raw_rgb, raw_sigma = mlp(samples_enc, viewdirs_enc)
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
            "rays_d": rays["rays_d"]
        }
        return ret

    def forward(self, rays, randomized, white_bkgd, near, far):
        ret = []
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

            else:
                t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
                t_vals, samples = helper.sample_pdf(
                    bins=t_mids,
                    weights=weights[..., 1:-1],
                    origins=rays["rays_o"],
                    directions=rays["rays_d"],
                    t_vals=t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                )
                mlp = self.fine_mlp

            samples_enc = helper.pos_enc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
            )
            viewdirs_enc = helper.pos_enc(rays["viewdirs"], 0, self.deg_view)
            raw_rgb, raw_sigma = mlp(samples_enc, viewdirs_enc)

            if self.noise_std > 0 and randomized:
                raw_sigma = raw_sigma + torch.rand_like(raw_sigma) * self.noise_std

            rgb = self.rgb_activation(raw_rgb)
            sigma = self.sigma_activation(raw_sigma)

            comp_rgb, acc, weights, depth = helper.volumetric_rendering(
                rgb,
                sigma,
                t_vals,
                rays["rays_d"],
                white_bkgd=white_bkgd,
            )

            ret.append((comp_rgb, acc, depth))

        return ret


class LitNeRF(LitModel):
    def __init__(
        self,
        hparams,
        lr_init: float = 5.0e-4,
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
        super(LitNeRF, self).__init__()
        self.model = NeRF()

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

    def model_forward(self, batch):
        
        seg_results = []
        rgb_c_results = []
        rgb_f_results = []
        # deform_c2ws = []
        part_num = 1
        w, h = self.hparams.img_wh
        pix_inds = torch.arange(0, w*h)
        # for part in range(part_num):

        #     # get part indicator
        #     part_indicator = F.one_hot(torch.tensor(part), self.hparams.part_num).reshape([1, -1])
        #     part_indicator = part_indicator.to(device=batch['img'].device, dtype=batch['img'].dtype)

        # view deformation
        # input: c2w + articulation + part indicator
        # apply rigid body transformation
        
        c2w = batch['c2w'].to(torch.float32)

        # deform_c2ws += [deform_c2w]

        # get_rays after deformation
        rays_o, viewdirs, rays_d = get_rays_torch(batch['directions'], c2w[:3, :], output_view_dirs=True)
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
            c_result = self.model.model_forward(batch_chunk, self.randomized, self.white_bkgd, self.near, self.far, corase_dict, i_level=0)

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
            f_result = self.model.model_forward(batch_chunk, self.randomized, self.white_bkgd, self.near, self.far, fine_ip_dict, i_level=1)

            f_rgb, f_acc, f_weights, f_depth = helper.volumetric_rendering(
                f_result["rgb"],
                f_result["sigma"],
                f_result["t_vals"],
                f_result["rays_d"],
                self.white_bkgd
                )
            frgb_results += [f_rgb]
    
        
        return torch.cat(frgb_results, dim=0)

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

        rendered_results = self.model(
            batch, self.randomized, self.white_bkgd, self.near, self.far
        )
        rgb_coarse = rendered_results[0][0]
        rgb_fine = rendered_results[1][0]
        target = batch["target"]

        loss0 = helper.img2mse(rgb_coarse, target)
        loss1 = helper.img2mse(rgb_fine, target)
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
        ret = self.render_rays(batch, batch_idx)
        # rank = dist.get_rank()
        rank = 0
        if rank == 0:
            if batch_idx == self.random_batch:
                grid_img = visualize_val_rgb_opa_depth((W, H), batch, ret)
                self.logger.experiment.log({"val/GT_pred rgb": wandb.Image(grid_img)})

        return self.render_rays(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        for k, v in batch.items():
            batch[k] = v.squeeze()
            if k == "radii":
                batch[k] = v.unsqueeze(-1)
            if k == "near_obj" or k == "far_obj":
                batch[k] = batch[k].unsqueeze(-1)
        # return self.render_rays_test(batch, batch_idx)
        ret = self.render_rays_forward_test(batch, batch_idx)
        
        ret["target"] = batch["target"]
        ret["instance_mask"] = batch["instance_mask"]
        return ret

    def configure_optimizers(self):
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
                "ckpts", self.hparams.exp_name, 'test'
            )
            os.makedirs(image_dir, exist_ok=True)
            img_list = store_image(image_dir, rgbs, "image")
            print('images are stored in: ', image_dir)
            result_path = os.path.join("ckpts", self.hparams.exp_name, "results.json")
            write_stats(result_path, psnr, ssim, lpips, psnr_obj)
            self.logger.log_image(key = "test/results", images = img_list)
        return psnr, ssim, lpips

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
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
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
        viewdirs /= torch.norm(viewdirs, dim=-1, keepdim=True)
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