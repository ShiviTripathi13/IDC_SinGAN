import torch
from SinGAN.models import Generator

def SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, masks=None, diversity=1):
    """
    Loads the generator state (assumed stored in Gs[0]) and
    produces a generated image from random noise.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = Generator(in_channels=3, nfc=opt.nfc, num_layers=opt.num_layer,
                     ker_size=opt.ker_size, padd_size=opt.padd_size).to(device)
    netG.load_state_dict(Gs[0])
    netG.eval()
    
    noise = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        fake = netG(noise)
    fake = (fake + 1) / 2  # Scale from [-1,1] to [0,1]
    return [fake.cpu().numpy()[0]]


def SinGAN_generate_multi(Gs, Zs, reals, NoiseAmp, opt, num_samples=1, diversity=1):
    """
    Generates a list of sample images by using different random noise vectors.
    The diversity parameter can be used in more advanced versions to control
    the variation between outputs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the generator and load the state from Gs
    netG = Generator(in_channels=3, nfc=opt.nfc, num_layers=opt.num_layer,
                     ker_size=opt.ker_size, padd_size=opt.padd_size).to(device)
    netG.load_state_dict(Gs[0])
    netG.eval()

    generated_images = []
    for i in range(num_samples):
        # Generate a new noise vector for diversity
        noise = torch.randn(1, 3, 256, 256, device=device)
        with torch.no_grad():
            fake = netG(noise)
        # Convert from [-1,1] to [0,1]
        fake = (fake + 1) / 2
        generated_images.append(fake.cpu().numpy()[0])
        print(f"Generated sample {i}: range min {fake.min().item()}, max {fake.max().item()}")
    return generated_images
