import os
import time
from datetime import datetime
import torch
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
import numpy as np

from config import get_arguments
import SinGAN.functions as functions
from SinGAN.training import train

if __name__ == '__main__':
    parser = get_arguments()
    opt = parser.parse_args()
    
    # Post-config adjustment if needed.
    opt = functions.post_config(opt)
    
    # Generate directory to save the model and samples.
    opt.dir2save = functions.generate_dir2save(opt)
    if not os.path.exists(opt.dir2save):
        os.makedirs(opt.dir2save)
    
    # Read the input image (normalized to [-1, 1])
    real = functions.read_image(opt)
    
    # Save the normalized input for debugging purposes.
    plt.imsave(os.path.join(opt.dir2save, 'input_debug.png'), ((real+1)/2).squeeze().permute(1,2,0).numpy())
    print("Saved normalized input image for debugging.")

    # Adjust scales (dummy for now)
    real = functions.adjust_scales2image(real, opt)
    
    # Load the mask (if provided)
    functions.load_mask(opt)
    
    # Optionally, if a mask exists, you might blend it with the image.
    if opt.mask_original_image is not None:
        opt.mask_original_image = (opt.mask_original_image > 0.5).float()
        masked_background = torch.tensor([0.0, -0.5, 0.0], device=real.device).view(1, 3, 1, 1)  # Green
        real_masked = opt.mask_original_image * real + (1 - opt.mask_original_image) * masked_background
        real = real_masked
        plt.imsave(os.path.join(opt.dir2save, 'masked_debug.png'), ((real + 1) / 2).squeeze().permute(1, 2, 0).numpy())
        plt.imsave('binary_mask_debug.png', opt.mask_original_image.squeeze().cpu().numpy(), cmap='gray')
        print("Using masked input for training with background filling.")
    
    # Check whether a pretrained model exists.
    trained_model_path = os.path.join(opt.dir2save, 'Gs.pth')
    if os.path.isfile(trained_model_path):
        print("Found trained model at", trained_model_path)
        Gs, Ds, Zs, reals, NoiseAmp, metrics = functions.load_trained_pyramid(opt)
    else:
        print("No trained model found. Starting training...")
        start_time = time.time()
        Gs, Ds, Zs, reals, NoiseAmp, metrics = train(opt, real)
        elapsed_time = time.time() - start_time
        print("Training completed in %.2f seconds." % elapsed_time)
    
    # Plot and save metrics like accuracy
    functions.plot_metrics(metrics, opt.dir2save)
    print("Metrics visualization saved.")

    print("Training finished. Check", opt.dir2save, "for trained model files.")