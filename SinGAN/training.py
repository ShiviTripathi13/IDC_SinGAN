import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from SinGAN.models import Generator, Discriminator
from SinGAN.functions import read_image

def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # For demonstration, we create a dummy image tensor.
    # In practice, you could pass functions.read_image(opt) from main_train.py.
    # image = torch.randn(1, 3, 256, 256, device=device)
    image = read_image(opt).to(device)
    
    netG = Generator(in_channels=3, nfc=opt.nfc, num_layers=opt.num_layer,
                     ker_size=opt.ker_size, padd_size=opt.padd_size).to(device)
    netD = Discriminator(in_channels=3, nfc=opt.nfc, num_layers=opt.num_layer,
                         ker_size=opt.ker_size, padd_size=opt.padd_size).to(device)
    
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    num_epochs = 100
    for epoch in range(num_epochs):
        # Update Discriminator
        optimizerD.zero_grad()
        fake = netG(image)
        real_out = netD(image)
        fake_out = netD(fake.detach())
        lossD = F.mse_loss(real_out, torch.ones_like(real_out)) + \
                F.mse_loss(fake_out, torch.zeros_like(fake_out))
        lossD.backward()
        optimizerD.step()
        
        # Update Generator
        optimizerG.zero_grad()
        fake = netG(image)
        fake_out = netD(fake)
        with torch.no_grad():
            output_range = (fake.min().item(), fake.max().item())
            print("Generator output range (before scaling):", output_range)
        lossG = F.mse_loss(fake_out, torch.ones_like(fake_out))
        lossG.backward()
        optimizerG.step()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Loss_D: {lossD.item():.4f}, Loss_G: {lossG.item():.4f}")
    
    # Save the generator pyramid state as Gs.pth into the save directory.
    save_path = os.path.join(opt.dir2save, 'Gs.pth')
    torch.save(netG.state_dict(), save_path)
    print("Saved Generator state to", save_path)
    
    # Return dummy values in order to keep compatibility with the rest of SinGAN.
    return [netG.state_dict()], None, None, None, None
