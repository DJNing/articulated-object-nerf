import nerfacc
from nerfacc.estimators.occ_grid import OccGridEstimator
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from models.vanilla_nerf import ngp_utils as ngp
from datasets import dataset_dict
from torch.utils.data import DataLoader
import os
from random import random
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models.interface import LitModel
import models.vanilla_nerf.helper as helper
import torchvision.transforms as T

'''
aabb: bounxing boxes for the object, examples: torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)

'''

class LitNeRFAcc(LitModel):
    def __init__(
        self,
        hparams,
        lr_init: float = 1.0e-1,
        lr_final: float = 5.0e-5,
        lr_delay_steps: int = 100,
        lr_delay_mult: float = 0.01,
        randomized: bool = True,
        lr_art: float = 1e-2,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__", "hparams"]:
                print(name, value)
                setattr(self, name, value)
        self.hparams.update(vars(hparams))
        super(LitNeRFAcc, self).__init__()
        self.hparams.white_back = False
        aabb = torch.Tensor(np.array(self.hparams.aabb))
        grid_res = self.hparams.grid_res
        grid_nlvl = self.hparams.grid_nlvl
        self.estimator = OccGridEstimator(roi_aabb=aabb, resolution=grid_res, levels=grid_nlvl)
        self.radience_field = ngp.NGPRadianceField(self.estimator.aabbs[-1])
    
    def setup(self, stage: Optional[str] = None) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]
        
        self.near = self.hparams.near
        self.far = self.hparams.far
        kwargs_train = {
            "root_dir": self.hparams.root_dir,
            "img_wh": tuple(self.hparams.img_wh),
            "white_back": self.hparams.white_back,
            "model_type": "vailla_nerf",
            "record_hard_sample": self.hparams.record_hard_sample,
            "near": self.hparams.near,
            "far": self.hparams.far,
            "use_keypoints": self.hparams.use_keypoints,
        }
        kwargs_val = {
            "root_dir": self.hparams.root_dir,
            "img_wh": tuple(self.hparams.img_wh),
            "white_back": self.hparams.white_back,
            "model_type": "vanilla_nerf",
            "near": self.hparams.near,
            "far": self.hparams.far,
        }
        
        if self.hparams.run_eval:
            kwargs_test = {
                "root_dir": self.hparams.root_dir,
                "img_wh": tuple(self.hparams.img_wh),
                "white_back": self.hparams.white_back,
                "model_type": "vanilla_nerf",
                "eval_inference": self.hparams.render_name,
                "near": self.hparams.near,
                "far": self.hparams.far,
            }
            self.test_dataset = dataset(split="test", **kwargs_test)
            self.near = self.test_dataset.near
            self.far = self.test_dataset.far
            self.white_bkgd = self.test_dataset.white_back

        else:
            self.train_dataset = dataset(split="train", **kwargs_train)
            self.val_dataset = dataset(split="val", **kwargs_val)
            self.white_bkgd = self.train_dataset.white_back

    def train_dataloader(self):
        self.__train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size=None,
            pin_memory=True,
        )
        return self.__train_loader

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
        
    def optimizer_step(self,
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
        pass
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.radience_field.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
            )
    
    def forward_all_samples(self, batch):
        
        pass
    
    def forward(self, batch):
        rays_o = batch['rays_o']
        rays_d = batch['rays_d']
        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            sigma= self.radience_field.query_density(positions)
            return sigma.squeeze(-1)
        
        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgbs, sigmas = self.radiance_field(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)
        
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            render_step_size=5e-3,
            stratified=self.radiance_field.training
        )
        
        rgb, opa, depth, extras = nerfacc.rendering(t_starts, t_ends, ray_indices, rgb_sigma_fn=rgb_sigma_fn)
        
        # ret_dict = {
        #     'rgb': rgb,
        #     'opa': opa,
        #     'depth': depth,
        #     'extras': extras
        # }
        return rgb, opa, depth, extras
    
    def training_step(self, batch, batch_idx):
        for k, v in batch.items():
            if k == "obj_idx" or k == 'idx':
                continue
            batch[k] = v.squeeze(0)
            
        rgb, opa, depth, extras = self.forward(batch)
        
        rgb_target = batch['rgb']
        
        rgb_loss = helper.img2mse(rgb, rgb_target)
        psnr = helper.mse2psnr(rgb_loss)
        self.log('train/rgb_loss', rgb_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/psnr', psnr, on_step=True, prog_bar=True, logger=True)
        
        return rgb_loss
    
    def validation_step(self, batch, batch_idx):
        for k, v in batch.items():
            if k == "obj_idx" or k == 'idx':
                continue
            batch[k] = v.squeeze(0)
            
        chunk = self.hparams.forward_chunk
        N = batch['rays_o'].shape[0]
        rgb_list = []
        depth_list = []
        for i in range(0, N, chunk):
            start_idx = i
            end_idx = min(i+chunk, N)
            mini_batch = {
                "rays_o": batch["rays_o"][start_idx:end_idx],
                "rays_d": batch["rays_d"][start_idx:end_idx],
            }
            rgb, opa, depth, extras = self.forward(batch)
            rgb_list += [rgb]
            depth_list += [depth]
        
        pred_rgb = torch.cat(rgb_list, dim=0)
        # pred_depth = torch.cat(depth_list, dim=0)
        W, H = self.hparams.img_wh
        def toPIL(tensor, h, w, c=3):
            img = tensor.view(h, w, c).permute(2, 0, 1).cpu()
            pil_img = T.ToPILImage()(img)
            return pil_img
        pred_img = toPIL(pred_rgb, H, W)
        # pred_depth_pil = toPIL(pred_depth, H, W, 1)
        gt = batch['rgb']
        mse = helper.img2mse(pred_rgb, gt)
        psnr = helper.mse2psnr(mse)
        gt_img = toPIL(gt, H, W)
        ret_dict = {
            'gt': gt_img,
            'pred': pred_img,
            'psnr': psnr
        }
        return ret_dict
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        psnr = sum(ret['psnr'] for ret in outputs) / len(outputs)
        self.log('val/psnr', psnr, on_epoch=True)
        gt_list = [ret['gt'] for ret in outputs]
        pred_list = [ret['pred'] for ret in outputs]
        self.logger.log_image(key='val/pred', images=pred_list)
        self.logger.log_image(key='val/gt', images=gt_list)
        return 
    
    def test_step(self, batch, batch_idx):
        pass