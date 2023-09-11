# %%
import torch
import torch.nn as nn
from models.vanilla_nerf.model_part_seg import DeformationMLP, NeRFSeg
from datasets.sapien import SapienPartDataset
from opt import get_parser
import json
from utils.partseg_utils import *
from datasets import dataset_dict
from models.vanilla_nerf.helper import mse2psnr, img2mse
import wandb
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# wandb_logger = WandbLogger()


def get_opts(args_list):

    parser = get_parser()

    args = parser.parse_args(args_list)
    # Load and parse the JSON configuration file
    with open(args.config, "r") as config_file:
        config_data = json.load(config_file)
        
    # Update the args namespace with loaded JSON data
    for key, value in config_data.items():
        setattr(args, key, value)
        
    return args

def load_nerf_weights(ckpt_path, model):
    ckpt_dict = torch.load(ckpt_path)
    load_dict = ckpt_dict['state_dict']
    new_ckpt_dict = remove_first_dot_in_keys(load_dict)
    model.load_state_dict(new_ckpt_dict)
    return


def remove_first_dot_in_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        # Split the key by dots and remove the first element
        new_key = '.'.join(key.split('.')[1:])
        new_state_dict[new_key] = value
    return new_state_dict

# %%
lr_init = 5.0e-4
epochs = 1
run = wandb.init(
    # Set the project where this run will be logged
    project="NeRF_part_seg_supervised",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr_init,
        "epochs": epochs,
    })
# %%
nerf = NeRFSeg()
ckpt_path = "results/official/nerf_official/last.ckpt"
load_nerf_weights(ckpt_path, nerf)
init_seg_layer = nn.Linear(128 + 2, 1)
comp_dim = nerf.num_coarse_samples + nerf.num_fine_samples + 1
comp_seg_layer = nn.Linear(comp_dim, 1)
criterion = nn.CrossEntropyLoss()
view_deform = DeformationMLP(8, 256, 15, 12)
seg_layers = [init_seg_layer, comp_seg_layer]

network_list = [nerf, init_seg_layer, comp_seg_layer, view_deform]
device = torch.device("cuda")
for i in network_list:
    i.to(device)

opt_params = []
for p in init_seg_layer.parameters():
    opt_params += [p]

for p in comp_seg_layer.parameters():
    opt_params += [p]

for p in view_deform.parameters():
    opt_params += [p]


# automatic_optimization = False
# %%
optimizer = torch.optim.Adam(
            params=iter(opt_params), lr=lr_init, betas=(0.9, 0.999)
        )
# set wandb log


# %%
arg_lists = ["--config", "config/partseg_train.json"]
opt = get_opts(arg_lists)
supervised = True

# %%
# get dataloader 

train_loader, val_loader = get_dataloader(opt)

# %%
def train(
        model, dataloader, num_epochs, optimizer, criterion, opts,\
              seg_layers, view_deform, logger, randomized=True, white_bkgd=True):
    
    near, far = dataloader.dataset.near, dataloader.dataset.far
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}', position=0, leave=True)
        for input_dict in dataloader:
            for k, v in input_dict.items():
                if k == "obj_idx":
                    continue
                input_dict[k] = v.squeeze(0).to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            for k, v in input_dict.items():
                if k == "obj_idx":
                    continue
                input_dict[k] = v.squeeze(0) # squeeze the batch_size dim

            seg_pred, pix_inds, c_rgbs, f_rgbs = \
                model_forawrd(input_dict, opts, model, seg_layers,\
                               view_deform, near, far, randomized, white_bkgd)
            
            
            if supervised:
                seg_final = F.softmax(seg_pred, dim=-1)
                seg_target = input_dict['seg_one_hot'][pix_inds][:, 1:] # remove background label
                seg_loss = criterion(seg_final, seg_target)
                
                rgb_target = input_dict['img'][pix_inds]
                # gather part rgb values
                
                c_pred, c_target = gather_part_rgb(seg_target, c_rgbs, rgb_target)
                f_pred, f_target = gather_part_rgb(seg_target, f_rgbs, rgb_target)

                loss0 = img2mse(c_pred, c_target)
                loss1 = img2mse(f_pred, f_target)

                psnr0 = mse2psnr(loss0)
                psnr1 = mse2psnr(loss1)

                loss = loss0 + loss1 + seg_loss

                log_dict = {
                    "seg_loss": seg_loss.item(),
                    "psnr1": psnr1.item(),
                    "psnr0": psnr0.item(),
                    "loss": loss.item()
                }
                # logger.log("train/seg_loss", seg_loss)

                # logger.log("train/psnr1", psnr1)
                # logger.log("train/psnr0", psnr0)
                # logger.log("train/loss", loss)
            
            
            # Update the progress bar description with the current loss
            progress_bar.set_description(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4e}, seg_loss: {seg_loss.item():.4e}, rgb_loss: {(loss0 + loss1).item():.4e}')
            
            # Update the progress bar
            progress_bar.update(1)
            # Backpropagation
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            running_loss += loss.item()
        progress_bar.close()
        # Print the average loss for this epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(dataloader)}')

# %%
torch.autograd.set_detect_anomaly(True)
train(nerf, train_loader, epochs, optimizer, criterion, opt, seg_layers,\
      view_deform, wandb)
