# %% [markdown]
# # iNeRF with segmentation

# %%
import torch
import torch.nn as nn
from models.vanilla_nerf.model_nerfseg import NeRFSeg
from models.vanilla_nerf.helper import load_state_dict_and_report
from PIL import Image
from pathlib import  Path as P
import json
import torchvision.transforms as transforms
import numpy as np
from datasets.ray_utils import get_ray_directions
import matplotlib.pyplot as plt
from models.vanilla_nerf.model_nerfseg import  get_rays_torch
from models.vanilla_nerf.helper import img2mse
# from pytorch3d.transforms import quaternion_to_matrix
import torch.nn.functional as F
from utils.viewpoint import pose2view_torch, view2pose_torch, change_apply_change_basis_torch

# %% [markdown]
# ## helper functions and classes

# %%
# helper functions
def load_dict_and_report(model, pretrained_state_dict):
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

            
from utils.rotation import *
from utils.viewpoint import *
class ArticulationEstimation(nn.Module):
    '''
    Current implemetation for revolute only
    '''
    def __init__(self, mode='qua') -> None:
        super().__init__()
        if mode == 'qua':
            pass
        elif mode == 'rad': #radian
            pass
        elif mode == 'deg': # degree
            pass
        else:
            raise RuntimeError('mode == %s for ArticulationEstimation is not defined' % mode)
        
        # perfect init
        # init_Q = torch.Tensor([ 0.9962,  0.0000, -0.0872,  0.0000])
        # axis_origin = torch.Tensor([ 0.24714715,  0.        , -0.00770604])
        # normal init
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
        E1 = view2pose_torch(c2w)
        translation_matrix = torch.eye(4).to(c2w)
        translation_matrix[:3, 3] = self.axis_origin.view([3])
        rotation_matrix = torch.eye(4).to(c2w)
        R = R_from_quaternions(self.Q)
        rotation_matrix[:3, :3] = R
        E2 = change_apply_change_basis_torch(E1, rotation_matrix, translation_matrix)
        view = pose2view_torch(E2)
        return view
def fetch_img(root_path, transform_meta, w=640, h=480, device='cuda', if_fix=True):
    if if_fix:
        idx = 5
    else:
        idx = np.random.randint(0, 9)
    frame_id = 'r_' + str(idx)
    pose_np = np.array(transform_meta['frame'][frame_id])

    rgb_pil = Image.open(str(root_path/'rgb'/(frame_id + '.png')))
    rgb = transforms.ToTensor()(rgb_pil).to(device)
    rgb = rgb.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
    rgb = rgb[:, :3]*rgb[:, -1:] + (1-rgb[:, -1:]) # blend A to RGB

    pose = torch.Tensor(pose_np).to(device)

    seg_pil = Image.open(str(root_path/'seg'/(frame_id + '.png')))
    seg_np = np.array(seg_pil)
    seg = torch.Tensor(seg_np).to(device).view([1, -1]).permute(1, 0)
    seg = seg.type(torch.LongTensor)
    seg = seg - 1 # starts with 2
    seg[seg<0] = 0
    focal = transform_meta['focal']
    directions = get_ray_directions(h, w, focal).view([-1, 3])
    mask = seg > 0
    ret_dict = {
        'rgb': rgb.to(device),
        'seg': seg.to(device),
        'directions': directions.to(device),
        'pose': pose.to(device),
        'mask': mask.to(device)
    }
    return ret_dict

def load_json(json_fname):
    with open(json_fname, 'r') as json_file:
        data_dict = json.load(json_file)
    return data_dict

# %% [markdown]
# ## Load NeRFSeg models with NeRF checkpoints

# %%
import sys
from opt import get_opts

sys.argv = ['', '--config', 'config/train_nerfseg.json']
device = 'cuda'
hparams= get_opts()
nerfseg = NeRFSeg(hparams)
ckpt = torch.load(hparams.nerf_ckpt)
load_dict = {}
length = len('model.')
state_dict = ckpt['state_dict']
for key in state_dict.keys():
    if key[:length] == 'model.':
        load_dict[key[length:]] = state_dict[key]

load_dict_and_report(nerfseg, load_dict)
# load_state_dict_and_report(nerfseg, hparams.nerf_ckpt)
nerfseg = nerfseg.to(device)
art_est = ArticulationEstimation().to(device)
near = 2
far = 6

# %% [markdown]
# ## Config optimizer

# %%
seg_params = []
for name, param in nerfseg.named_parameters():
    if 'seg' in name:
        param.requires_grad = True
        seg_params += [param]
    else:
        param.requires_grad = False

art_params = []
for _, param in art_est.named_parameters():
    param.requires_grad = True
    art_params += [param]

art_lr = 1e-2
seg_lr = 1e-3

seg_opt_dict = {
    'params': seg_params,
    'lr': seg_lr
}

art_opt_dict = {
    'params': art_params,
    'lr': art_lr
}

optimizer = torch.optim.Adam([seg_opt_dict, art_opt_dict], lr=seg_lr)


# %%
from tqdm import tqdm
optimize_step = 1e3
result = []
data_root = P("./data/laptop_art_same_pose/train/idx_5/")
transform_meta = load_json(str(data_root / 'transforms.json'))
ray_chunk_size = 4096
data_dict = fetch_img(data_root, transform_meta)

directions = data_dict['directions']
rgb = data_dict['rgb']
seg = data_dict['seg']
pose = data_dict['pose']
mask = data_dict['mask']
random_indx = torch.randint(0, directions.shape[0], [ray_chunk_size])
random_dirs = directions[random_indx]
random_rgbs = rgb[random_indx]
random_mask = mask[random_indx]

# %%
new_pose = art_est(pose)
rays_o, viewdirs, rays_d = get_rays_torch(random_dirs, new_pose[:3, :], output_view_dirs=True)
# gather input_dict for NeRF
input_dict = {
    'rays_o': rays_o,
    'rays_d': rays_d,
    'viewdirs': viewdirs
}

num_rays = rays_o.shape[0]
part_code = torch.zeros([1, hparams.part_num]).to(rays_o)
part_code[:, int(1)] = 1
input_dict['part_code'] = part_code
rendered_results = nerfseg(input_dict, False, True, near, far)


# %%
coarse_dict = rendered_results['level_0']
fine_dict = rendered_results['level_1']

# %%


# %%



