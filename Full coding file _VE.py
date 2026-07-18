# test_pipeline.py

import torch
import numpy as np
from main import *

def test_vae():
    """Test VAE module"""
    print("Testing VAE...")
    vae = VAE(latent_dim=128)
    x = torch.randn(8, 1, 256, 256)
    recon, mu, logvar = vae(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Mu shape: {mu.shape}")
    print(f"  Logvar shape: {logvar.shape}")
    
    # Test sampling
    samples = vae.sample(4)
    print(f"  Samples shape: {samples.shape}")
    print("  ✅ VAE test passed!")

def test_gan():
    """Test GAN module"""
    print("Testing GAN...")
    generator = GANGenerator()
    discriminator = PatchGANDiscriminator()
    
    x = torch.randn(8, 1, 256, 256)
    fake = generator(x)
    pred = discriminator(fake)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Generated shape: {fake.shape}")
    print(f"  Discriminator output shape: {pred.shape}")
    print("  ✅ GAN test passed!")

def test_attention_unet():
    """Test Attention U-Net"""
    print("Testing Attention U-Net...")
    model = AttentionUNet()
    
    x = torch.randn(8, 1, 256, 256)
    pred = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {pred.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  ✅ Attention U-Net test passed!")

if __name__ == "__main__":
    print("="*50)
    print("Running Tests")
    print("="*50)
    test_vae()
    test_gan()
    test_attention_unet()
    print("\n✅ All tests passed!")