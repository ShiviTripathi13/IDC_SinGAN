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
    # For debugging, you can comment out mask blending to see if training improves.
    if opt.mask_original_image is not None:
        # For completion tasks, you could combine: known regions from the mask and unknown filled with the image.
        # Here, we assume mask values close to 1 mean “keep original” while 0 means “to be completed.”
        real_masked = opt.mask_original_image * real
        real = real_masked
        print("Using masked input for training.")
    
    # Check whether a pretrained model exists.
    trained_model_path = os.path.join(opt.dir2save, 'Gs.pth')
    if os.path.isfile(trained_model_path):
        print("Found trained model at", trained_model_path)
        # For simplicity in this example, we load the state via functions.load_trained_pyramid.
        Gs, Ds, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    else:
        print("No trained model found. Starting training...")
        start_time = time.time()
        Gs, Ds, Zs, reals, NoiseAmp = train(opt, real)
        elapsed_time = time.time() - start_time
        print("Training completed in %.2f seconds." % elapsed_time)
    
    print("Training finished. Check", opt.dir2save, "for trained model files.")
