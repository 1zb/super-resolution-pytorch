import torch.nn as nn
import torch.nn.functional as F

def srcnn(f1=9, f2=1, f3=5, n1=64, n2=32):
    model = nn.Sequential(
        nn.Conv2d(1, n1, kernel_size=f1, stride=1, padding=(f1-1)//2, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(n1, n2, kernel_size=f2, stride=1, padding=0, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(n2, 1, kernel_size=f3, stride=1, padding=(f3-1)//2, bias=False),
    )
    return model

def fsrcnn(d=56, s=12, m=4, upscale_factor=2):
    layers = [
        nn.Conv2d(1, d, kernel_size=5, stride=1, padding=2, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(d, s, kernel_size=1, stride=1, padding=0, bias=False),
        nn.ReLU(inplace=True),
    ]
    for _ in range(m):
        layers.append(nn.Conv2d(s, s, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Conv2d(s, d, kernel_size=1, stride=1, padding=0, bias=False))
    layers.append(nn.ReLU(inplace=True))

    if upscale_factor == 2:
        layers.append(nn.ConvTranspose2d(d, 1, kernel_size=9, stride=2, padding=4, output_padding=1, bias=False))
    elif upscale_factor == 3:
        layers.append(nn.ConvTranspose2d(d, 1, kernel_size=9, stride=3, padding=3, bias=False))
    elif upscale_factor == 4:
        layers.append(nn.ConvTranspose2d(d, 1, kernel_size=9, stride=4, padding=3, output_padding=1, bias=False))
    else:
         raise NotImplementedError('Only supported 2x and 3x!')

    model = nn.Sequential(*layers)
    return model

class ExtReconBlock(nn.Module):
    def __init__(self, upscale_factor=2, depth=5, channels=64):
        super(ExtReconBlock, self).__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        if upscale_factor == 2:
            layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1, bias=False))
        elif upscale_factor == 3:
            layers.append(nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=3, padding=1, bias=False))
        else:
             raise NotImplementedError('Only supported 2x and 3x!')
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.ext = nn.Sequential(*layers)

        if upscale_factor == 2:
            self.upscale = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        elif upscale_factor == 3:
            self.upscale = nn.ConvTranspose2d(1, 1, kernel_size=5, stride=3, padding=1, bias=False)
        else:
            raise NotImplementedError('Only supported 2x and 3x!')

        self.residual = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input_features, input_image):
        output_features = self.ext(input_features)
        output_image = self.upscale(input_image) + self.residual(output_features)
        return output_features, output_image

class LapSRN(nn.Module):
    def __init__(self, scale_levels=[2,2,2], depth=5, channels=64):
        super(LapSRN, self).__init__()

        self.scale_levels = scale_levels

        self.conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False)

        for i in range(len(self.scale_levels)):
            self.add_module('level{}'.format(i), ExtReconBlock(upscale_factor=self.scale_levels[i], depth=depth, channels=channels))

    def forward(self, *images):
        assert len(images) == len(self.scale_levels)
        features = self.conv(images[0])

        recons = []
        for i in range(len(images)):
            features, hr = getattr(self, 'level{}'.format(i))(features, images[i])
            recons.append(hr)

        return tuple(recons)

def lapsrn(**kwargs):
    model = LapSRN(**kwargs)
    return model

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
    def forward(self, input):
        residual = self.residual(input)
        return residual + input

class SRResNet(nn.Module):
    def __init__(self, in_channels=3, ngf=64, scale_levels=[2,2], res_depth=16):
        super(SRResNet, self).__init__()

        ## TODO change kernel_size of conv_pre and conv_end to 9
        
        self.conv_pre = nn.Conv2d(in_channels, ngf, kernel_size=3, stride=1, padding=1, bias=False)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(ngf) for i in range(res_depth)],
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
        )

        layers = []
        for scale in scale_levels:
            layers.append(nn.Conv2d(ngf, ngf * scale * scale, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.PixelShuffle(scale))
            layers.append(nn.ReLU(inplace=True))

        self.upscale = nn.Sequential(*layers)

        self.conv_end = nn.Conv2d(ngf, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input):
        x = F.relu(self.conv_pre(input))
        x = x + self.res_blocks(x)
        x = upscale(x)
        return F.tanh(self.conv_end(x))

class SRGANDis(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super(SRGANDis, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.main(input)
        x = x.view(x.size(0), 512 * 6 * 6)
        return self.classifier(x)
