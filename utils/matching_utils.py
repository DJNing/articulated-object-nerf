from kornia.feature import LoFTR
import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from pathlib import Path as P

import matplotlib
import matplotlib.pyplot as plt
def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig
    
def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    # x = 1 - np.clip(err / (thr * 2), 0, 1)
    x = err / (thr *2)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)

def show_LoFTR_matching(fname1, fname2, thr=0.5):
    img1 = cv2.cvtColor(cv2.imread(fname1), cv2.COLOR_RGB2GRAY)/255
    img2 = cv2.cvtColor(cv2.imread(fname2), cv2.COLOR_RGB2GRAY)/255
    w, h = img1.shape
    img1_tensor = torch.tensor(img1).reshape(1, 1, w, h).to(torch.float)
    img2_tensor = torch.tensor(img2).reshape(1, 1, w, h).to(torch.float)
    input = {"image0": img1_tensor, "image1": img2_tensor}
    loftr = LoFTR('indoor')
    out = loftr(input)
    mkps0 = out['keypoints0']
    mkps1 = out['keypoints1']
    confidence = out['confidence']
    mkps0 = out['keypoints0'][confidence>thr]
    mkps1 = out['keypoints1'][confidence>thr]
    conf = confidence[confidence>thr]

    color = error_colormap(conf, thr)
    fig = make_matching_figure(img1, img2, mkps0, mkps1, color)
    return fig, mkps0, mkps1

def show_LoFTR_matching_topK(fname1, fname2, thr=1e-4, topK=50):
    img1 = cv2.cvtColor(cv2.imread(fname1), cv2.COLOR_RGB2GRAY)/255
    img2 = cv2.cvtColor(cv2.imread(fname2), cv2.COLOR_RGB2GRAY)/255
    w, h = img1.shape
    img1_tensor = torch.tensor(img1).reshape(1, 1, w, h).to(torch.float)
    img2_tensor = torch.tensor(img2).reshape(1, 1, w, h).to(torch.float)
    input = {"image0": img1_tensor, "image1": img2_tensor}
    loftr = LoFTR('indoor')
    out = loftr(input)
    mkps0 = out['keypoints0']
    mkps1 = out['keypoints1']
    confidence = out['confidence']
    try:
        _, topK_idx = torch.topk(confidence, topK)
        topK_conf = confidence[topK_idx]
        mkps0 = mkps0[topK_idx, :]
        mkps1 = mkps1[topK_idx, :]
    except:
        topK_conf = confidence
        
    color = error_colormap(topK_conf, thr)
    fig = make_matching_figure(img1, img2, mkps0, mkps1, color)
    return fig, mkps0, mkps1

def create_video_from_images(image_list, output_filename, fps=30):
    # Define the codec and create a VideoWriter object
    init_image = cv2.imread(str(image_list[0]))
    H, W, _ = init_image.shape
    frame_size = (W, H)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D') # save avi only
    out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    # Iterate through the image list and add them to the video
    for image_path in tqdm(image_list):
        image = cv2.imread(str(image_path))
        
        # Resize image to match frame size
        # image_resized = cv2.resize(image, frame_size)
        
        # Write the image to the video
        out.write(image)

    # Release the VideoWriter and close the video file
    out.release()
    
def get_loftr_matching_vid(src_path, match_path, vid_fname, fps=15, topK=50, skip_gen=False):
    images = sorted(list(src_path.glob('*.png')))
    match_path.mkdir(exist_ok=True)
    
    if not skip_gen:
    # draw matching figures
        for i in tqdm(range(len(images) - 1)):
            img_0 = str(images[i])
            img_1 = str(images[i+1])
            fig, _, _ = show_LoFTR_matching_topK(img_0, img_1, topK=topK)
            fname = str(i).zfill(4) + '.png'
            save_name = match_path / fname
            fig.savefig(str(save_name))
            plt.close()
        
    # generate videos
    vid_imgs = sorted(list(match_path.glob("*.png")))
    create_video_from_images(vid_imgs, vid_fname, fps=fps)
    