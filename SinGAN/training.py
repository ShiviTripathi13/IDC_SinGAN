import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from SinGAN.models import Generator, Discriminator

def total_variation(x):
    """
    Total variation loss to reduce noise.
    x is expected to be of shape [batch, channels, height, width].
    """
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w

def train(opt, real):
    """
    Train the SinGAN generator and discriminator.
    `real` is the input image tensor (already normalized to [-1, 1]).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real = real.to(device)

    netG = Generator(in_channels=3, nfc=opt.nfc, num_layers=opt.num_layer,
                     ker_size=opt.ker_size, padd_size=opt.padd_size).to(device)
    netD = Discriminator(in_channels=3, nfc=opt.nfc, num_layers=opt.num_layer,
                         ker_size=opt.ker_size, padd_size=opt.padd_size).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    num_epochs = 2000  # Increase the number of epochs for better convergence
    lambda_tv = 0.2   # Weight for total variation loss
    
    for epoch in range(num_epochs):
        # =================== Train Discriminator ======================
        optimizerD.zero_grad()
        fake = netG(real)
        real_out = netD(real)
        fake_out = netD(fake.detach())
        lossD = F.mse_loss(real_out, torch.ones_like(real_out)) + \
                F.mse_loss(fake_out, torch.zeros_like(fake_out))
        lossD.backward()
        optimizerD.step()
        
        # =================== Train Generator ======================
        optimizerG.zero_grad()
        fake = netG(real)
        fake_out = netD(fake)
        lossG_adv = F.mse_loss(fake_out, torch.ones_like(fake_out))
        lossG_tv = lambda_tv * total_variation(fake)
        
        lossG = lossG_adv + lossG_tv
        lossG.backward()
        optimizerG.step()

        if epoch % 50 == 0:
          output_path = os.path.join(opt.dir2save, f'output_epoch_{epoch}.png')
          plt.imsave(output_path, ((fake[0].cpu().detach().numpy() + 1) / 2).transpose(1, 2, 0))
          print(f"Saved intermediate output image to {output_path}")
        
        # Log training progress every 50 epochs.
        if epoch % 50 == 0:
            with torch.no_grad():
                out_min = fake.min().item()
                out_max = fake.max().item()
            print(f"Epoch [{epoch}/{num_epochs}] Loss_D: {lossD.item():.4f}, Loss_G: {lossG_adv.item():.4f}+TV: {lossG_tv.item():.4f}, Output range: [{out_min:.4f}, {out_max:.4f}]")
    
    # Save the generator's state (the pyramid, here simplified, as Gs.pth)
    save_path = os.path.join(opt.dir2save, 'Gs.pth')
    torch.save(netG.state_dict(), save_path)
    print("Saved Generator state to", save_path)
    
    # For compatibility, return a tuple (Gs, Ds, Zs, reals, NoiseAmp)
    # Here, only Gs is relevant.
    return [netG.state_dict()], None, None, None, None
