"""
Cetak torchinfo.summary BISINDOClassifier untuk Bab 4.3.1
Jalankan: python backend/model_summary.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.mobilenetv2 import InvertedResidual

# ── salin definisi model dari train_v10.py ─────────────────────────────────

class TemporalShift(nn.Module):
    def __init__(self, n_segment=16, n_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div  = n_div

    def forward(self, x):
        BT, C, H, W = x.size()
        B = BT // self.n_segment
        x = x.view(B, self.n_segment, C, H, W)
        fold = C // self.fold_div
        out = torch.zeros_like(x)
        out[:, 1:,    :fold]     = x[:, :-1, :fold]
        out[:, :-1, fold:2*fold] = x[:, 1:,  fold:2*fold]
        out[:, :,   2*fold:]     = x[:, :,   2*fold:]
        return out.view(BT, C, H, W)


def inject_tsm(model, n_segment=16, n_div=8):
    for block in model.features:
        if isinstance(block, InvertedResidual):
            orig = block.conv[0]
            block.conv[0] = nn.Sequential(TemporalShift(n_segment, n_div), orig)
    return model


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(nn.Linear(ch, ch//r, bias=False), nn.ReLU(),
                                  nn.Linear(ch//r, ch, bias=False))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        a = self.fc(self.avg(x).view(B, C))
        m = self.fc(self.mx(x).view(B, C))
        return x * self.sig(a + m).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.channel_att = ChannelAttention(ch, r)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


class BISINDOClassifier(nn.Module):
    def __init__(self, num_classes=24, num_frames=16,
                 grid=4, reduce_dim=256, dropout=0.5):
        super().__init__()
        self.num_frames = num_frames
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        backbone = inject_tsm(backbone, n_segment=num_frames, n_div=8)
        self.features = backbone.features
        feat_dim      = backbone.last_channel   # 1280

        self.cbam   = CBAM(feat_dim)
        self.reduce = nn.Sequential(
            nn.Conv2d(feat_dim, reduce_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True))
        self.pool        = nn.AdaptiveAvgPool2d((grid, grid))
        self.spatial_dim = reduce_dim * grid * grid   # 256*4*4 = 4096

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.spatial_dim, num_classes))

    def forward(self, x):   # (B, T, 3, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.cbam(x)
        x = self.reduce(x)
        x = self.pool(x).view(B, T, -1)
        x = x.mean(dim=1)
        return self.classifier(x)


# ── main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        from torchinfo import summary
    except ImportError:
        print('[ERROR] torchinfo belum terinstall. Jalankan:')
        print('        pip install torchinfo')
        sys.exit(1)

    print('Membangun model...')
    model = BISINDOClassifier(num_classes=24, num_frames=16)
    model.eval()

    print('\n' + '='*70)
    result = summary(
        model,
        input_size=(1, 16, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=3,
        verbose=1
    )
    print('='*70)

    total   = sum(p.numel() for p in model.parameters())
    trainbl = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal parameter      : {total:,}')
    print(f'Trainable parameter  : {trainbl:,}')
    print(f'Non-trainable param  : {total - trainbl:,}')
