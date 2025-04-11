import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import get_arguments
import SinGAN.functions as functions
from SinGAN.manipulate import SinGAN_generate

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    opt = parser.parse_args()
    
    opt.dir2save = functions.generate_dir2save(opt)
    trained_model_path = os.path.join(opt.dir2save, 'Gs.pth')
    if not os.path.isfile(trained_model_path):
        print("No trained model found. Please run main_train.py first.")
        exit()
    
    # Load the trained pyramid (here, only the generator state)
    Gs, Ds, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    
    opt.mode = 'random_samples'
    opt.gen_start_scale = 0
    images = SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
    sample = images[0]
    print("Generated sample value range: min =", sample.min(), "max =", sample.max())

    
    for idx, img in enumerate(images):
        sample_path = os.path.join(opt.dir2save, f'random_sample_{idx}.png')
        # Transpose from CxHxW to HxWxC for saving using matplotlib
        plt.imsave(sample_path, np.transpose(img, (1, 2, 0)))
    print("Random samples saved in", opt.dir2save)
