import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


seg_colormap = torch.tensor([
        [255, 0, 0],   # Red
        [0, 255, 0],   # Green
        [0, 0, 255],   # Blue
        [255, 255, 0], # Yellow
        [255, 0, 255], # Magenta
        [0, 255, 255], # Cyan
        [128, 0, 0],   # Maroon
        [0, 128, 0],   # Green (dark)
        [0, 0, 128],   # Blue (dark)
        [128, 128, 128] # Gray
    ], dtype=torch.uint8)

def visualize_depth(depth, cmap=cv2.COLORMAP_JET, vmin=None, vmax=None):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    mi = np.min(x) if vmin == None else vmin  # get minimum depth
    ma = np.max(x) if vmax == None else vmax
    x = np.clip(x, mi, ma)
    x = (x - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_


def visualize_val_image(img_wh, batch, results, typ="fine"):
    W, H = img_wh
    rgbs = batch["rgbs"]
    img_inst = (
        results[f"rgb_instance_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()
    )  # (3, H, W)
    img_full = results[f"rgb_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    # img_bg = results[f'rgb_bg_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    # if batch["depths"].sum() == 0:
    #     vis_min, vis_max = None, None
    # else:
    #     vis_min, vis_max = (
    #         batch["depths"].min().item() + 0.3,
    #         batch["depths"].max().item(),
    #     )
    vis_min, vis_max = None, None
    depth_inst = visualize_depth(
        results[f"depth_instance_{typ}"].view(H, W), vmin=vis_min, vmax=vis_max
    )  # (3, H, W)
    depth = visualize_depth(
        results[f"depth_{typ}"].view(H, W), vmin=vis_min, vmax=vis_max
    )  # (3, H, W)
    # gt_depth = visualize_depth(batch["depths"].view(H, W), vmin=vis_min, vmax=vis_max)
    opacity = visualize_depth(
        results[f"opacity_instance_{typ}"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)
    # stack = torch.stack(
    #     [img_gt, img_inst, img_full, depth_inst, depth, gt_depth, opacity]
    # )  # (4, 3, H, W)
    stack = torch.stack(
        [img_gt, img_inst, img_full, depth_inst, depth, opacity]
    )  # (6, 3, H, W)
    grid = make_grid(stack, nrow=3)
    img = T.ToPILImage()(grid)
    return img


def visualize_val_image_instance(img_wh, batch, results, typ="fine"):
    W, H = img_wh

    crop_img = False
    if crop_img:
        W -= 2 * 32
        H -= 2 * 32
    rgbs = batch["rgbs"]
    img_inst = (
        results[f"rgb_instance_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()
    )  # (3, H, W)
    # img_full = results[f"rgb_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    # img_bg = results[f'rgb_bg_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    # if batch["depths"].sum() == 0:
    #     vis_min, vis_max = None, None
    # else:
    #     vis_min, vis_max = (
    #         batch["depths"].min().item() + 0.3,
    #         batch["depths"].max().item(),
    #     )
    vis_min, vis_max = None, None
    depth_inst = visualize_depth(
        results[f"depth_instance_{typ}"].view(H, W), vmin=vis_min, vmax=vis_max
    )  # (3, H, W)
    # depth = visualize_depth(
    #     results[f"depth_{typ}"].view(H, W), vmin=vis_min, vmax=vis_max
    # )  # (3, H, W)
    # gt_depth = visualize_depth(batch["depths"].view(H, W), vmin=vis_min, vmax=vis_max)
    opacity = visualize_depth(
        results[f"opacity_instance_{typ}"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)
    # stack = torch.stack(
    #     [img_gt, img_inst, img_full, depth_inst, depth, gt_depth, opacity]
    # )  # (4, 3, H, W)
    stack = torch.stack([img_gt, img_inst, depth_inst, opacity])  # (6, 3, H, W)
    grid = make_grid(stack, nrow=2)
    img = T.ToPILImage()(grid)
    return img


def visualize_val_opacity(img_wh, batch, results):
    W, H = img_wh
    opacity = visualize_depth(
        results["acc"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)
    target_mask = visualize_depth(
        batch["instance_mask"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)
    stack = torch.stack([target_mask, opacity])  # (6, 3, H, W)
    grid = make_grid(stack, nrow=1)
    img = T.ToPILImage()(grid)
    return img


def visualize_val_rgb(img_wh, batch, results):
    W, H = img_wh

    rgbs = batch["target"]
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    pred_rgb = results["comp_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    stack = torch.stack([img_gt, pred_rgb])  # (6, 3, H, W)
    grid = make_grid(stack, nrow=1)
    img = T.ToPILImage()(grid)
    return img

def reverse_one_hot(a):
    return torch.argmax(a, dim=-1)

def generate_seg_color_maps(a, b, K):
    # Assuming you have tensors a and b
    # a = reverse_one_hot(a)
    # b = reverse_one_hot(b)


    # Create a mask for values equal to 0
    mask_a = (a == 0)
    mask_b = (b == 0)

    # Apply the mask to set color_map_a and color_map_b to [0, 0, 0] where a and b are 0
    color_map_a = seg_colormap[a]
    color_map_a = color_map_a * (~mask_a[..., None])  # Apply mask to each channel

    color_map_b = seg_colormap[b]
    color_map_b = color_map_b * (~mask_b[..., None])  # Apply mask to each channel

    return color_map_a, color_map_b

def visualize_val_rgb_seg(img_wh, batch, results):
    W, H = img_wh

    rgbs = batch["target"]
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    pred_rgb = results["comp_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    stack = torch.stack([img_gt, pred_rgb])  # (6, 3, H, W)

    seg_target = batch['seg_one_hot'].view(H, W, 3).cpu()
    seg_pred = results['comp_seg'].view(H, W, 3).cpu()

    seg_target = reverse_one_hot(seg_target)
    seg_pred = reverse_one_hot(seg_pred)

    seg_target, seg_pred = generate_seg_color_maps(seg_target, seg_pred, K=10)
    seg_pred = seg_pred.permute(2, 0 ,1).to(torch.float32) / 255
    seg_target = seg_target.permute(2, 0 ,1).to(torch.float32) / 255
    stack = torch.stack([img_gt, pred_rgb, seg_target, seg_pred]) 
    grid = make_grid(stack, nrow=2)
    img = T.ToPILImage()(grid)
    return img


def visualize_val_rgb_opa_depth(img_wh, batch, results):
    W, H = img_wh

    rgbs = batch["target"]
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    pred_rgb = results["comp_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()

    vis_min, vis_max = None, None
    depth = visualize_depth(
        results["depth"].view(H, W), vmin=vis_min, vmax=vis_max
    )  # (3, H, W)

    opacity = visualize_depth(
        results["acc"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)

    stack = torch.stack([img_gt, pred_rgb, depth, opacity])  # (6, 3, H, W)
    grid = make_grid(stack, nrow=2)
    img = T.ToPILImage()(grid)
    return img


def visualize_val_obj_fb_bg_rgb(img_wh, batch, results):
    W, H = img_wh

    rgbs = batch["target"]
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    pred_rgb = results["comp_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()

    pred_obj_rgb = results["obj_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_fg_rgb = results["fg_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_bg_rgb = results["bg_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    stack = torch.stack(
        [img_gt, pred_rgb, pred_obj_rgb, pred_fg_rgb, pred_bg_rgb]
    )  # (6, 3, H, W)
    grid = make_grid(stack, nrow=2)
    img = T.ToPILImage()(grid)
    return img


def visualize_val_fb_bg_rgb(img_wh, batch, results):
    W, H = img_wh

    rgbs = batch["target"]
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    pred_rgb = results["comp_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_fg_rgb = results["fg_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_bg_rgb = results["bg_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()

    stack = torch.stack([img_gt, pred_rgb, pred_fg_rgb, pred_bg_rgb])  # (6, 3, H, W)
    grid = make_grid(stack, nrow=2)
    img = T.ToPILImage()(grid)
    return img


def visualize_val_fb_bg_rgb_opacity(img_wh, batch, results):
    W, H = img_wh

    rgbs = batch["target"]
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    pred_rgb = results["comp_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_obj_rgb = results["obj_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_fg_rgb = results["fg_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_bg_rgb = results["bg_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()

    opacity = visualize_depth(
        results["obj_acc"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)
    target_mask = visualize_depth(
        batch["instance_mask"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)

    stack = torch.stack(
        [img_gt, pred_rgb, pred_obj_rgb, pred_fg_rgb, pred_bg_rgb, target_mask, opacity]
    )  # (6, 3, H, W)
    grid = make_grid(stack, nrow=2)
    img = T.ToPILImage()(grid)
    return img


def visualize_val_rgb_opacity_nocs(img_wh, batch, results):
    W, H = img_wh

    rgbs = batch["target"]
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    pred_rgb = results["comp_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_obj_rgb = results["obj_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_nocs = results["nocs"].view(H, W, 3).permute(2, 0, 1).cpu()
    gt_nocs = batch["nocs_2d"].view(H, W, 3).permute(2, 0, 1).cpu()

    opacity = visualize_depth(
        results["obj_acc"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)
    target_mask = visualize_depth(
        batch["instance_mask"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)

    stack = torch.stack(
        [img_gt, pred_rgb, pred_obj_rgb, target_mask, opacity, pred_nocs, gt_nocs]
    )  # (6, 3, H, W)
    grid = make_grid(stack, nrow=3)
    img = T.ToPILImage()(grid)
    return img


def visualize_val_fb_bg_rgb_opacity_nocs(img_wh, batch, results):
    W, H = img_wh

    rgbs = batch["target"]
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    pred_rgb = results["comp_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    if "obj_rgb" in results:
        pred_obj_rgb = results["obj_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_fg_rgb = results["fg_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_bg_rgb = results["bg_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    pred_nocs = results["nocs"].view(H, W, 3).permute(2, 0, 1).cpu()
    gt_nocs = batch["nocs_2d"].view(H, W, 3).permute(2, 0, 1).cpu()

    opacity = visualize_depth(
        results["obj_acc"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)
    target_mask = visualize_depth(
        batch["instance_mask"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)

    if "obj_rgb" in results:
        stack = torch.stack(
            [
                img_gt,
                pred_rgb,
                pred_obj_rgb,
                pred_fg_rgb,
                pred_bg_rgb,
                target_mask,
                opacity,
                pred_nocs,
                gt_nocs,
            ]
        )  # (6, 3, H, W)
        grid = make_grid(stack, nrow=3)
    else:
        stack = torch.stack(
            [
                img_gt,
                pred_rgb,
                pred_fg_rgb,
                pred_bg_rgb,
                target_mask,
                opacity,
                pred_nocs,
                gt_nocs,
            ]
        )  # (6, 3, H, W)
        grid = make_grid(stack, nrow=2)

    img = T.ToPILImage()(grid)
    return img


def visualize_val_rgb_opacity(img_wh, batch, results):
    W, H = img_wh

    rgbs = batch["target"]
    print("rgbs", rgbs.shape)
    img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    print("results[comp_rgb]", results["comp_rgb"].shape)
    pred_rgb = results["comp_rgb"].view(H, W, 3).permute(2, 0, 1).cpu()
    opacity = visualize_depth(
        results["acc"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)
    target_mask = visualize_depth(
        batch["instance_mask"].unsqueeze(-1).view(H, W),
        vmin=0,
        vmax=1,
    )  # (3, H, W)
    stack = torch.stack([img_gt, pred_rgb, target_mask, opacity])  # (6, 3, H, W)
    grid = make_grid(stack, nrow=2)
    img = T.ToPILImage()(grid)
    return img
