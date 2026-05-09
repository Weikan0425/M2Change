import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from utils.simvp.net import SimVP_Model
from utils.stdiff import *
from utils.head import UPerHead

def eau(x,patch_size):
    b, t, c, h, w = x.shape

    avg_channel = x.mean(dim=2, keepdim=True)

    x_3ch = torch.cat([x, avg_channel], dim=2)

    x_3ch = x_3ch.view(b * t, 3, h, w)

    x_up = F.interpolate(x_3ch, size=(patch_size, patch_size), mode="bilinear", align_corners=False)

    x_up = x_up.view(b, t, 3, patch_size, patch_size)

    return x_up

class TDS(nn.Module):
    """
    输入: (B, T, C, H, W)
    输出: (B, C, H, W)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        last = x[:, -1]                 # (B, C, H, W)
        prev = x[:, :-1]                # (B, T-1, C, H, W)
        # 计算绝对差
        diff = (prev - last.unsqueeze(1)).abs()  # (B, T-1, C, H, W)
        # 沿时间维度求和
        out = diff.sum(dim=1)           # (B, C, H, W)
        return out
    
class AuxRefine(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.refine_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, d, aux):

        B, C, H, W = d.shape

        aux_resized = F.interpolate(aux, size=(H, W), mode='bilinear', align_corners=False)

        mask = F.softmax(aux_resized, dim=1)[:, 1, :, :]  # (B, H, W)
        mask = mask.unsqueeze(1)  # (B, 1, H, W)
        mask = nn.Sigmoid()(mask)  # (B, 1, H, W)

        d_refined = d * mask
        if self.use_conv:
            d_refined = self.refine_conv(d_refined)

        d_refined = d + d_refined
        return d_refined

class M2CDViT(nn.Module):
    def __init__(self, num_classes = 2, n_times = 12, image_size = 40, t_dim = 128, pretrain = True) -> None:
        super(M2CDViT, self).__init__()
        self.image_size = image_size
        
        self.pair_encoder = timm.create_model('convnext_base', pretrained = pretrain, features_only=True)
        self.video_encoder = SimVP_Model(in_shape=(n_times, 3, image_size, image_size), out_dim=t_dim, model_type='convnext')
        
        decoder_dim = [128, 256, 512, 1024]
        # decoder_dim = [96, 192, 384, 768]

        self.stdiff_1 = STDiff(in_channel=decoder_dim[0], dim=t_dim, patch_size=4)
        self.stdiff_2 = STDiff(in_channel=decoder_dim[1], dim=t_dim, patch_size=2)
        self.stdiff_3 = STDiff(in_channel=decoder_dim[2], dim=t_dim, patch_size=2)
        self.stdiff_4 = STDiff(in_channel=decoder_dim[3], dim=t_dim, patch_size=1)

        self.tds = TDS()
        self.aux_1 = AuxRefine(channels=t_dim, use_conv=True)
        self.aux_2 = AuxRefine(channels=t_dim, use_conv=True)
        self.aux_3 = AuxRefine(channels=t_dim, use_conv=True)
        self.aux_4 = AuxRefine(channels=t_dim, use_conv=True)

        self.decoder = UPerHead(in_channels=[t_dim, t_dim, t_dim, t_dim], num_classes=num_classes, channels=t_dim)
        self.aux_head = nn.Conv2d(t_dim, num_classes, kernel_size=1)

    def forward(self, hr, s1):

        # video encoder tn features
        s1 = eau(s1, patch_size=self.image_size)
        s1_token = self.video_encoder(s1)
        aux = self.aux_head(self.tds(s1_token))

        # data preparation
        t1 = hr.clone()
        t2 = s1[:,-1,:,:,:]
        t2 = t2.squeeze(1)
        t2 = F.interpolate(t2, size=hr.shape[-2:], mode='bilinear', align_corners=True)

        # pair encoder t1 t2 features
        x1, x2, x3, x4 = self.pair_encoder(t1)
        y1, y2, y3, y4 = self.pair_encoder(t2)

        # stdiff features
        d1, w1 = self.stdiff_1(x1, y1, s1_token)
        d2, w2 = self.stdiff_2(x2, y2, s1_token)
        d3, w3 = self.stdiff_3(x3, y3, s1_token)
        d4, w4 = self.stdiff_4(x4, y4, s1_token)

        # aux refine
        d1 = self.aux_1(d1, aux)
        d2 = self.aux_2(d2, aux)
        d3 = self.aux_3(d3, aux)
        d4 = self.aux_4(d4, aux)

        # decoder
        out = self.decoder([d1, d2, d3, d4])

        out = F.interpolate(out, size=hr.shape[-2:], mode='bilinear', align_corners=True)
        aux = F.interpolate(aux, size=hr.shape[-2:], mode='bilinear', align_corners=True)
        w = [w1, w2, w3, w4]

        return out, aux, w

if __name__ == '__main__':

    model = M2CDViT().cuda()
    model_path = "projects/M2Change/dlcodes/MCD-m2cd/M2CDViT.pth"
    
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    x = torch.randn(2, 3, 400, 400).cuda()  # 输入示例
    y = torch.randn(2, 12, 2, 40, 40).cuda()
    with torch.no_grad(): 
        out, aux , w = model(x, y)  
    print(out.shape)

    from thop import profile
    flops, params = profile(model, inputs=(x,y), verbose=False)
    print("FLOPs: {:.3f} G".format(flops / 1e9))
    print("Params (all): {:.3f} M".format(params / 1e6))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: {:.3f} M".format(trainable_params / 1e6))