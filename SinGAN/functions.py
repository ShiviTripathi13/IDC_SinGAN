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
    If --input_mask is provided, loads the mask from Input/Images/ as a grayscale tensor.
    """
    if opt.input_mask != "":
        mask_path = os.path.join(opt.input_dir, 'Images', opt.input_mask)
        if os.path.isfile(mask_path):
            img = Image.open(mask_path).convert('L')
            transform = transforms.Compose([transforms.ToTensor()])
            mask_tensor = transform(img).unsqueeze(0)  # shape: [1, 1, H, W]
            opt.mask_original_image = mask_tensor
            print("Loaded mask with range:", mask_tensor.min().item(), mask_tensor.max().item())
        else:
            print("Mask file not found at", mask_path)
            opt.mask_original_image = None
    else:
        opt.mask_original_image = None

def calc_scale_to_start_masking(opt, real):
    """Returns a dummy scale index for starting masking."""
    return 2
