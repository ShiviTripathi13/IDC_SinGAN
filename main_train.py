import os
import time
from datetime import datetime
import torch
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt

from config import get_arguments
import SinGAN.functions as functions
from SinGAN.training import train

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    
    # Post-process configuration if needed.
    opt = functions.post_config(opt)
    
    # Determine the directory to save the trained model
    opt.dir2save = functions.generate_dir2save(opt)
    if not os.path.exists(opt.dir2save):
        os.makedirs(opt.dir2save)
    
    # Read the input image and adjust scales
    real = functions.read_image(opt)
    real = functions.adjust_scales2image(real, opt)
    
    # Optionally load mask if provided
    functions.load_mask(opt)
    
    # Check if a trained model exists
    trained_model_path = os.path.join(opt.dir2save, 'Gs.pth')
    if os.path.isfile(trained_model_path):
        print("Found trained model at", trained_model_path)
        Gs, Ds, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    else:
        print("No trained model found. Starting training...")
        start_time = time.time()
        Gs, Ds, Zs, reals, NoiseAmp = train(opt)
        elapsed_time = time.time() - start_time
        print("Training completed in %.2f seconds." % elapsed_time)
    
    # Optionally save the mask image if loaded
    if opt.mask_original_image is not None:
        plt.imsave(os.path.join(opt.dir2save, 'mask.png'), opt.mask_original_image.cpu().squeeze(), cmap='gray')
    
    print("Training finished. Check", opt.dir2save, "for trained model files.")
