class Pydnet(nn.Module):
    """
    Implements PydNet (https://arxiv.org/abs/1806.11430) in Pytorch.
    
    For mobilePydnet (https://arxiv.org/abs/2006.05724), use mobile_version = True
    """
    def __init__(self, mobile_version = False):
        super(Pydnet, self).__init__()
        
        self.channels = [16, 32, 64, 96, 128, 192]
        
        # Define all layers       
        for i in range(len(self.channels)): # 0-5
            # Encoder pyramid
            setattr(self, "conv{}".format(i), self.conv_block(3 if i==0 else self.channels[i-1], self.channels[i]))
        
            # Decoders
            setattr(self, "decoder{}".format(i), self.decoder_block(self.channels[i] if i==5 else self.channels[i]+8)) # +8 for the concatenated layers from previous output
            
        # Upsampling
        if mobile_version:
            # Transpose convolutions have been replaced by upsampling and convolution blocks to avoid checkerboard artifacts (https://distill.pub/2016/deconv-checkerboard/)
            self.deconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(8, 8, kernel_size=2, stride=1)
            )
        else:
            self.deconv = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2)
        
    def forward(self, x):
        # Pass through encoder
        conv_out0 = self.conv0(x)
        conv_out1 = self.conv1(conv_out0)
        conv_out2 = self.conv2(conv_out1)
        conv_out3 = self.conv3(conv_out2)
        conv_out4 = self.conv4(conv_out3)
        conv_out5 = self.conv5(conv_out4)
        
        # L6: scale 5
        out5 = self.decoder5(conv_out5)
        deconv_out5 = self.deconv(out5)
        disp5 = self.get_disp(out5, mobile_version)
        conv_out4 = torch.cat((conv_out4, deconv_out5), 1)
        # L5: scale 4
        out4 = self.decoder4(conv_out4)
        deconv_out4 = self.deconv(out4)
        disp4 = self.get_disp(out4, mobile_version)
        conv_out3 = torch.cat((conv_out3, deconv_out4), 1)
        # L4: scale 3
        out3 = self.decoder3(conv_out3)
        deconv_out3 = self.deconv(out3)
        disp3 = self.get_disp(out3, mobile_version)
        conv_out2 = torch.cat((conv_out2, deconv_out3), 1)
        # L3: scale 2
        out2 = self.decoder2(conv_out2)
        deconv_out2 = self.deconv(out2)
        disp2 = self.get_disp(out2, mobile_version)
        conv_out1 = torch.cat((conv_out1, deconv_out2), 1)
        # L2: scale 1
        out1 = self.decoder1(conv_out1)
        deconv_out1 = self.deconv(out1)
        disp1 = self.get_disp(out1, mobile_version)
        conv_out0 = torch.cat((conv_out0, deconv_out1), 1)
        # L1: scale 0
        out0 = self.decoder0(conv_out0)
        deconv_out0 = self.deconv(out0)
        disp0 = self.get_disp(out0, mobile_version)
        
        return [disp0, disp1, disp2, disp3, disp4, disp5]
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
    
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
    
    def get_disp(self, x, mobile_version=False):
        # Final activation regressor after each decoder
        if mobile_version:
            return nn.Conv2d(x.shape[1], 1, kernel_size=3, stride=1, padding=1)(x)
        else:
            return 0.3*nn.Sigmoid()(x[:,-1,:,:])
            