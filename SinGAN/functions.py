import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def read_image(opt):
    """Reads the input image from Input/Images/."""
    path = os.path.join(opt.input_dir, 'Images', opt.input_name)
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # Shape: [1, 3, H, W]
    return img_tensor

def adjust_scales2image(real, opt):
    """
    In a full SinGAN, the input image is resized for multiple scales.
    Here, for simplicity we just set the stop_scale.
    """
    opt.stop_scale = opt.num_scales
    return real

def generate_dir2save(opt):
    """Generates a directory to save the trained model and output samples."""
    base_name = os.path.splitext(opt.input_name)[0]
    dir2save = os.path.join("TrainedModels", base_name)
    if not os.path.exists(dir2save):
        os.makedirs(dir2save)
    return dir2save

def load_trained_pyramid(opt):
    """
    Dummy function that loads the trained generator pyramid from 'Gs.pth'.
    Returns: (Gs, Ds, Zs, reals, NoiseAmp). For simplicity we only load Gs.
    """
    dir2save = generate_dir2save(opt)
    gs_path = os.path.join(dir2save, 'Gs.pth')
    if os.path.exists(gs_path):
        state = torch.load(gs_path, map_location=torch.device("cpu"))
        return [state], None, None, None, None
    else:
        return None, None, None, None, None

def post_config(opt):
    """Any post-processing of options can be done here."""
    return opt

def load_mask(opt):
    """If a mask is provided (e.g. church_mask.png), load it."""
    mask_path = os.path.join(opt.input_dir, 'Images', opt.input_mask) if hasattr(opt, 'input_mask') and opt.input_mask != '' else ''
    if mask_path and os.path.isfile(mask_path):
        img = Image.open(mask_path).convert('L')  # Load in grayscale
        transform = transforms.Compose([transforms.ToTensor()])
        mask_tensor = transform(img).unsqueeze(0)
        opt.mask_original_image = mask_tensor
    else:
        opt.mask_original_image = None

def calc_scale_to_start_masking(opt, real):
    """
    Dummy function that returns a scale index at which to start applying the mask.
    """
    return 2
