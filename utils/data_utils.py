import torch 
import torch.nn.functional as F

def sliding_window_crop_batched(img, window_size=(256, 256), stride=(172, 172)):
    #default to window size 256x256, stride 172
    #so that get 9 patches from 512x512 image with overlap.
    """
    Crop an image tensor into overlapping patches using a sliding window.
    Returns a single batched tensor of shape [N, C, H_win, W_win].

    Args:
       img: torch.Tensor or np.ndarray
             Shape (C,H,W) or (H,W,C)
        window_size: (int,int)
             Crop height and width
        stride: (int,int)
             Vertical and horizontal stride

    Returns:
        patches: torch.Tensor of shape [N, C, H_win, W_win]
        positions: list of (y,x) top-left pixel coordinates
    """
    if not isinstance(img, torch.Tensor):
        img = torch.from_numpy(img)

    if img.ndim == 2:
        img = img.unsqueeze(0)                   # -> [1,H,W]
    elif img.shape[-1] <= 4:                     # likely HWC
        img = img.permute(2, 0, 1)               # -> [C,H,W]

    c, h, w = img.shape
    win_h, win_w = window_size
    s_h, s_w = stride

    # unfold creates sliding blocks without explicit loops
    patches = img.unfold(1, win_h, s_h).unfold(2, win_w, s_w)  # [C, nH, nW, win_h, win_w]
    nH, nW = patches.shape[1:3]

    # rearrange to batch dimension
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()      # [nH, nW, C, win_h, win_w]
    patches = patches.view(-1, c, win_h, win_w)                # [N, C, win_h, win_w]

    # compute top-left coordinates for each patch
    positions = [(y * s_h, x * s_w) for y in range(nH) for x in range(nW)]

    return patches, positions


def make_sr_lr_pair(batch_x: torch.Tensor, scale: int = 2):
    """
    Generate (LR, SR) triplets by bicubic downsampling and re-upscaling.

    Args:
        batch_x (torch.Tensor): Input tensor [N, C, H, W] — high-res batch.
        scale (int): Downsampling factor.

    Returns:
        batch_y (torch.Tensor): Bicubic downsampled batch (low-res).
        batch_y_up (torch.Tensor): Bicubic upsampled batch (reconstructed SR baseline).
    """
    if batch_x.ndim != 4:
        raise ValueError("Input must be [N, C, H, W].")

    N, C, H, W = batch_x.shape
    h_lr, w_lr = H // scale, W // scale

    # Bicubic downsampling
    batch_y = F.interpolate(batch_x, size=(h_lr, w_lr), mode="bicubic", align_corners=False)

    # Bicubic upsampling back to original size
    batch_y_up = F.interpolate(batch_y, size=(H, W), mode="bicubic", align_corners=False)

    return batch_y_up, batch_y