import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block with Up-Sampling and Skip Connections.
    """
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ECGModel(nn.Module):
    """
    ResNet34 Encoder + U-Net Decoder for ECG Signal Segmentation.
    """
    def __init__(self, backbone='resnet34.a3_in1k', num_classes=4, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        # Get channel counts from the backbone automatically
        enc_channels = self.encoder.feature_info.channels() 
        
        # Decoder Configuration (High to Low channels)
        # We reverse the encoder channels for the skip connections
        self.dec4 = DecoderBlock(enc_channels[-1], enc_channels[-2], 256)
        self.dec3 = DecoderBlock(256, enc_channels[-3], 128)
        self.dec2 = DecoderBlock(128, enc_channels[-4], 64)
        self.dec1 = DecoderBlock(64, enc_channels[-5], 64) # Assuming 5 stages
        
        # Final pixel-wise classification head
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        features = self.encoder(x) # Returns list of features [f1, f2, f3, f4, f5]
        
        # Decoder Path (with skips)
        d = self.dec4(features[-1], features[-2])
        d = self.dec3(d, features[-3])
        d = self.dec2(d, features[-4])
        d = self.dec1(d, features[-5])
        
        # Final Upscale to match input resolution (approx)
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
        
        return self.head(d)

if __name__ == "__main__":
    # Sanity Check
    model = ECGModel(pretrained=False)
    dummy_input = torch.randn(2, 3, 512, 1024) # Batch, Ch, H, W
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}") # Should be [2, 4, 512, 1024]