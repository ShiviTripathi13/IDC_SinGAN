import torch
from SinGAN.models import Generator

def SinGAN_generate_multi(Gs, Zs, reals, NoiseAmp, opt, num_samples=1, diversity=1):
    """
    Generates multiple samples using different random noise inputs.
    Loads the generator state from Gs[0].
    Returns a list of generated images (numpy arrays with shape [3, H, W]).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(in_channels=3, nfc=opt.nfc, num_layers=opt.num_layer,
                     ker_size=opt.ker_size, padd_size=opt.padd_size).to(device)
    netG.load_state_dict(Gs[0])
    netG.eval()
    
    generated_images = []
    for i in range(num_samples):
        noise = torch.randn(1, 3, 256, 256, device=device)
        with torch.no_grad():
            fake = netG(noise)
        # Scale output from [-1, 1] to [0, 1]
        fake = (fake + 1) / 2
        generated_images.append(fake.cpu().numpy()[0])
        print(f"Sample {i} generated with range: min = {fake.min().item()}, max = {fake.max().item()}")
    return generated_images
