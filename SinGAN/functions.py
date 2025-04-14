import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def read_image(opt):
    """Reads the input image from Input/Images/ and normalizes to [-1, 1]."""
    path = os.path.join(opt.input_dir, 'Images', opt.input_name)
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, H, W]
    # Normalize from [0, 1] to [-1, 1]
    img_tensor = img_tensor * 2 - 1
    return img_tensor

def adjust_scales2image(real, opt):
    """
    Dummy function for multi-scale adjustment.
    Sets opt.stop_scale equal to opt.num_scales.
    """
    opt.stop_scale = opt.num_scales
    return real

def generate_dir2save(opt):
    """Generates and returns a directory path to save trained models and outputs."""
    base_name = os.path.splitext(opt.input_name)[0]
    dir2save = os.path.join("TrainedModels", base_name)
    if not os.path.exists(dir2save):
        os.makedirs(dir2save)
    return dir2save

def load_trained_pyramid(opt):
    """
    Loads the trained generator state from Gs.pth.
    Returns a tuple (Gs, Ds, Zs, reals, NoiseAmp).
    For simplicity, only Gs is loaded.
    """
    dir2save = generate_dir2save(opt)
    gs_path = os.path.join(dir2save, 'Gs.pth')
    if os.path.exists(gs_path):
        state = torch.load(gs_path, map_location=torch.device("cpu"))
        return [state], None, None, None, None
    else:
        return None, None, None, None, None

def post_config(opt):
    """Returns the post-processed configuration (if needed)."""
    return opt

def load_mask(opt):
    """
    Loads a grayscale mask from Input/Images/ and normalizes it to [0, 1].
    If --input_mask is provided, it loads the mask into opt.mask_original_image.
    """
    if opt.input_mask != "":
        mask_path = os.path.join(opt.input_dir, 'Images', opt.input_mask)
        if os.path.isfile(mask_path):
            img = Image.open(mask_path).convert('L')  # Load as grayscale
            transform = transforms.Compose([transforms.ToTensor()])
            mask_tensor = transform(img).unsqueeze(0)  # shape: [1, 1, H, W]
            opt.mask_original_image = mask_tensor
            print("Loaded mask with range:", mask_tensor.min().item(), mask_tensor.max().item())
        else:
            print("Mask file not found at", mask_path)
            opt.mask_original_image = None
    else:
        opt.mask_original_image = None

def resize_and_tresh_mask(mask_tensor, scale):
    """
    Resizes and thresholds the mask to ensure it is binary.
    Args:
        mask_tensor: The original mask tensor (grayscale).
        scale: The scaling factor for resizing the mask.
    Returns:
        A resized binary mask tensor.
    """
    # Resize the mask
    mask_resized = torch.nn.functional.interpolate(mask_tensor, scale_factor=scale, mode='bilinear', align_corners=False)
    # Threshold the mask to make it binary
    mask_binary = (mask_resized > 0.5).float()
    return mask_binary

def creat_reals_mask_pyramid(real, reals, opt, mask):
    """
    Creates a pyramid of reals and corresponding masks for multi-scale SinGAN.
    Args:
        real: The input image tensor.
        reals: The list of real images at different scales.
        opt: Configuration options.
        mask: The initial mask tensor.
    Returns:
        Updated reals and masks pyramids.
    """
    reals.append(real)
    masks = [mask]
    for i in range(1, opt.stop_scale + 1):
        scale = pow(opt.scale_factor, opt.stop_scale - i)
        reals.append(torch.nn.functional.interpolate(real, scale_factor=scale, mode='bilinear', align_corners=False))
        masks.append(resize_and_tresh_mask(mask, scale))
    return reals, masks