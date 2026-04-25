

import torch
import torch.nn as nn
class SDFA(nn.Module):

    def __init__(self, in_channels, reduction=8, kernel_size=7):
        super(SDFA, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.kernel_size = kernel_size

        self.sal_avg_pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.sal_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        self.deg_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.deg_spatial_proj = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x_mean = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)

        mean_sq = self.sal_avg_pool(x_mean**2)
        sq_mean = self.sal_avg_pool(x_mean) ** 2
        variance = torch.clamp(mean_sq - sq_mean, min=1e-6)
        std_dev = torch.sqrt(variance)

        M_sal_raw = self.sal_conv(std_dev)  # (B, 1, H, W)

        fft_x = torch.fft.rfft2(x, norm="backward")
        mag_x = torch.abs(fft_x)  

        spectral_feat = torch.mean(torch.log(mag_x + 1e-8), dim=(2, 3), keepdim=True)

        W_freq = self.deg_fc(spectral_feat)  # (B, C, 1, 1)

        F_deg_weighted = x * W_freq
        M_deg_raw = self.deg_spatial_proj(F_deg_weighted)  # (B, 1, H, W)

        M_final = self.sigmoid(M_sal_raw + M_deg_raw)

        return x * M_final
    

if __name__ == '__main__':
    model = SDFA(in_channels=64, reduction=8, kernel_size=7)
    output = model(torch.randn(3, 64, 128, 128))
