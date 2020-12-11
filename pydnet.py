#
# MIT License
#
# Copyright (c) 2020 Zeeshan Khan Suri
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adapted from Pydnet TensorFlow model with
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it

import torch
from torch import nn
import torchvision
from torch.nn import functional as F

class SigmoidOnLastChannel(nn.Module):
    # Applies sigmoid on last channel of the input tensor and returns it
    def __init__(self):
        super(SigmoidOnLastChannel, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return 0.3 * self.sigmoid(x[:,-2:-1,:,:])

class Pydnet(nn.Module):
    # Pydnet wrapper to output in the same format as Monodepth2
    def __init__(self, scales=[0,1,2,3],
                 enc_version = "mobilepydnet", 
                 dec_version = "mobilepydnet",
                 pretrained=False):
        """
        Init:   
            scales:list(int): The scales for multi-scale output (default: [0,1,2,3])
            enc_version:string: Encoder model version (default: mobilepydnet)
                                    if contains pydnet: PydnetEncoder
                                        if also contain mobile: mobile version
                                Also supports timm models. Check rwightman.github.io. Requires timm library
                                e.g. if version="resnet18", loads resnet18 from timm library
            dec_version:string: Decoder model version (default: mobilepydnet)
                                    if contains pydnet: PydnetEncoder
                                        if also contain mobile: mobile version
                                    else is set to "general", general decoder is used
            pretrained:bool: Only valid for loading timm encoder models
        """
        super(Pydnet, self).__init__()

        self.scales = scales
        self.enc_version, self.dec_version = enc_version, dec_version
        
        # Encoder 
        if "pydnet" in enc_version:
            self.enc_channels = [16, 32, 64, 96, 128, 192]
            self.encoder = PydnetEncoder(mobile_version=True
                                            if "mobile" in enc_version else False,
                                         channels=self.enc_channels)
        else:
            import timm
            try:
                self.encoder = timm.create_model(enc_version,features_only=True,pretrained=pretrained)
                self.enc_channels = self.encoder.feature_info.channels()
            except:
                raise NotImplementedError("{} encoder not yet supported!".format(enc_version))
        # Decoder
        if "pydnet" in dec_version:
            self.decoder = PydnetDecoder(mobile_version=True if "mobile" in dec_version else False,
                                        channels=self.enc_channels,
                                        scales=self.scales)
        else:
            self.decoder = GeneralDecoder(channels=self.enc_channels,scales=self.scales)

    def forward(self, x):
        self.features=self.encoder(x)
        output = self.decoder(self.features)
        if "pydnet" in self.dec_version:
            return output
        else:
            return {k:F.interpolate(v,
                    scale_factor=2, mode = "bilinear", align_corners = True)
                         for k,v in output.items()}

class PydnetEncoder(nn.Module):
    def __init__(self, mobile_version=True, channels=[16, 32, 64, 96, 128, 192]) :
        super(PydnetEncoder, self).__init__()

        self.mobile_version = mobile_version
        self.channels = channels

        # Define all layers
        for i in range(len(self.channels)): # 0-5
            # Encoder pyramid
            setattr(self, "conv{}".format(i), self.conv_block(3 if i==0 else self.channels[i-1], self.channels[i]))
        
    def forward(self, x):
        # Pass through encoder
        features = []
        # x = (x - 0.45) / 0.225 # Normalize only for models pretrained on ImageNet

        for i in range(len(self.channels)):
            x = getattr(self, "conv{}".format(i))(x) 
            features.append(x)

        return features

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

class PydnetDecoder(nn.Module):
    """
    PydnetDecoder
    """
    def __init__(self, channels=[16, 32, 64, 96, 128, 192], scales=[0,1,2,3], mobile_version = False):
        super(PydnetDecoder, self).__init__()
        
        self.channels = channels
        self.scales = scales
        self.mobile_version = mobile_version
        
        # Define all layers
        for i in range(len(self.channels)): # 0-5
            # Decoders
            setattr(self, "decoder{}".format(i), self.decoder_block(self.channels[i] if i==len(self.channels)-1 else self.channels[i]+8)) # +8 for the concatenated layers from previous output
        
        # Final activation regressor after each decoder
        if mobile_version:
            self.regressor = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        else:
            self.regressor = SigmoidOnLastChannel()
        
        # Upsampling
        if mobile_version:
            # Transpose convolutions have been replaced by upsampling and convolution blocks to avoid checkerboard artifacts (https://distill.pub/2016/deconv-checkerboard/)
            self.deconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode = "bilinear", align_corners = True),
                # Original Tensorflow code uses 2x2 filter with same padding. For doing this in Pytorch, we need to do padding only on the right/bottom side of the image
                nn.ZeroPad2d((0,1,0,1)),
                nn.Conv2d(8, 8, kernel_size=2, stride=1, padding=0)
            )
        else:
            self.deconv = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2)

        
    def forward(self, features):
        # Pass through encoder
        outputs={}
        assert len(features)==len(self.channels), "# features must be same as len of channels" 
        x = features[-1]
        for i in range(len(self.channels)-2,-1,-1):
            x = getattr(self, "decoder{}".format(i+1))(x)
            deconv_out = self.deconv(x)
            if i+1 in self.scales:
                outputs[("disp",i+1)] = self.regressor(x)
            # if i>0:
            x = torch.cat((features[i], deconv_out), 1)

        x = self.decoder0(x)
        outputs[("disp",0)] = self.regressor(x)
        return outputs

    def decoder_block(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1),
        )

class GeneralDecoder(nn.Module):
    def __init__(self, channels=[64, 64, 128, 256, 512], scales=[0,1,2,3]):
        super(GeneralDecoder, self).__init__()

        self.scales = scales
        self.channels = channels

        # Define all layers
        for i in range(len(self.channels)-2,-1,-1): # 3,2,1,0
            # Decoders
            setattr(self, "decoder{}".format(i), DecoderBlock(self.channels[i+1],
                                                                    self.channels[i]))
            setattr(self, "regressor{}".format(i),
                    nn.Sequential(nn.Conv2d(self.channels[i],1,3,stride=1,padding=1),
                    nn.Sigmoid()))
        
        self.upsample = lambda x : F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, features):
        outputs={}
        assert len(features)==len(self.channels), "# features must be same as len of channels" 
        
        x = self.upsample(features[-1]) # latent feature
        for i in range(len(features)-2,-1,-1): # 3,2,1,0
            x = getattr(self, "decoder{}".format(i))(x)
            x += features[i] # u-net skip connections from encoder
            x = self.upsample(x)
            if i in self.scales:
                outputs[("disp", i)]= getattr(self, "regressor{}".format(i))(x)
        
        return outputs

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.decoder_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
                        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
    def forward(self, x):
        return self.decoder_block(x) + self.skip(x)


def decoder_block(self, in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
                        )

class PyddepthInference(Pydnet):
    def __init__(self, scales=[0,1,2,3], enc_version = "mobilepydne",dec_verison="mobile_pydnet", pretrained=False):
        super(PyddepthInference, self).__init__(scales, enc_version=enc_version, dec_version=dec_version)

        if pretrained:
            if enc_version=="":
                # Fetch pretrained Kitti model
                try:
                    loaded_dict = torch.hub.load_state_dict_from_url("https://github.com/zshn25/Pydnet-Pytorch/releases/download/v1.0/mobile_pydnet.pth")
                    new_dict = {}
                    for k in loaded_dict.keys():
                        new_dict[k.replace("pydnet.", "")] = loaded_dict[k]
                    self.pydnet.load_state_dict(new_dict)
                except:
                    print("Loading pretrained model failed. Please load it manually")
            else:
                raise NotImplementedError("Loading pretrained model failed. Pretrained model not available")

    @torch.no_grad()
    def forward(self, x):
        self.encoder.eval()
        self.decoder.eval()
        x=self.encoder(x)
        return self.decoder(x)[("disp",0)]
        #return F.interpolate(self.pydnet(x)[0], scale_factor=2, mode = "bilinear", align_corners = True)

