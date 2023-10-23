""""Baseline model for 3D U-Net is created here!

The model has 14.8M trainable parameters, slightly more than the original EF model.

Import this model in the main script and use it as follows:
    from UNet3D_baseline import UNet3D

...afterwards, initialize the model as follows:
    self.torch_nn_module = UNet3D(in_channels=3, out_channels=1)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a single convolutional block
def conv3d_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

# Define the Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.encode = nn.Sequential(
            conv3d_block(in_channels, out_channels),
            conv3d_block(out_channels, out_channels)
        )
        self.pool = nn.MaxPool3d(2, 2)
        
    def forward(self, x):
        x = self.encode(x)
        return x, self.pool(x)

# Define the Decoder block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.decode = nn.Sequential(
            conv3d_block(out_channels*2, out_channels),
            conv3d_block(out_channels, out_channels)
        )
        
    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], axis=1)
        x = self.decode(x)
        return x

class UNet3D(nn.Module): # That's my baby!
    '''Input: [b, 24, 192, 192, 3]
    output: [b, 24, 384, 384, 1]'''
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 256)
        
        # Decoder
        self.dec4 = DecoderBlock(256, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        
        # Final Layer
        self.final = nn.Conv3d(64, out_channels, kernel_size=1)
        
        # Additional Upsampling (spatial only)
        self.upsample = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
    def forward(self, x):
        # First thing should be permuting the channel orders!
        x = x.permute(0, 4, 1, 2, 3)
        
        # Encoder
        enc1, enc1_pool = self.enc1(x)
        enc2, enc2_pool = self.enc2(enc1_pool)
        enc3, enc3_pool = self.enc3(enc2_pool)
        enc4, _ = self.enc4(enc3_pool)
        
        # Decoder
        dec4 = self.dec4(enc4, enc3)
        dec3 = self.dec3(dec4, enc2)
        dec2 = self.dec2(dec3, enc1)
        
        # Final layer
        out = self.final(dec2)
        
        # Additional Upsampling (spatial only)
        out = self.upsample(out)
        
        # Permute to [b, time, height, width, channel]
        out = out.permute(0, 2, 3, 4, 1)
        
        return out