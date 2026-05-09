import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP2D(nn.Module):
    def __init__(self, channels, expansion=4, drop=0.):
        super(MLP2D, self).__init__()
        hidden_dim = int(channels * expansion)
        self.fc1 = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        self.drop = nn.Dropout(drop)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CAAM(nn.Module):
    def __init__(self, channels=128, expansion=4, drop=0.0):
        super(CAAM, self).__init__()
        self.mlpA = MLP2D(channels, expansion=expansion, drop=drop)
        self.mlpB = MLP2D(channels, expansion=expansion, drop=drop)

    def compute_cosine_similarity(self, xA, xB):
        xA_flat = xA.permute(0, 2, 3, 1).reshape(-1, xA.size(1))
        xB_flat = xB.permute(0, 2, 3, 1).reshape(-1, xB.size(1))
        cosine_sim = F.cosine_similarity(xA_flat, xB_flat, dim=1)
        return cosine_sim.view(xA.size(0), xA.size(2), xA.size(3))  # (B,H,W)

    def forward(self, xA, xB):

        c_weights = self.compute_cosine_similarity(xA, xB).unsqueeze(1)  # (B,1,H,W)
        c_weights = 1 - F.sigmoid(c_weights)

        xA_weighted = xA * c_weights
        xB_weighted = xB * c_weights

        xA_d = self.mlpA(xA_weighted)
        xB_d = self.mlpB(xB_weighted)

        outA = xA_d + xA
        outB = xB_d + xB

        return outA, outB, c_weights

class STAtt(nn.Module):
    def __init__(self, dim, t_dim, num_heads=8, patch_size=4):
        super().__init__()
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.dim = dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(t_dim, dim)
        self.v_proj = nn.Linear(t_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _extract_patches_2d(self, x):

        B, C, H, W = x.shape
        p = self.patch_size

        patches = F.unfold(x, kernel_size=p, stride=p)
        patches = patches.transpose(1, 2)  # [B, N, C*p*p]
        patches = patches.view(B, patches.shape[1], C, p, p)  # [B, N, C, p, p]

        patches = patches.mean(dim=[3,4])  # [B, N, C]
        return patches

    def _extract_patches_3d(self, x):

        B, T, C, H, W = x.shape
        p = self.patch_size
        patches_all = []
        for t in range(T):
            xt = x[:, t]  # [B, C, H, W]
            patches = self._extract_patches_2d(xt)  # [B, N, C]
            patches_all.append(patches)
        patches_all = torch.cat(patches_all, dim=1)  # [B, T*N, C]
        return patches_all

    def forward(self, q, kv):

        B, C, rHq, rWq = q.shape
        B, T, Ck, Hk, Wk = kv.shape

        q_patches = self._extract_patches_2d(q)   # [B, Nq, C]
        kv_patches = self._extract_patches_3d(kv) # [B, Nk, C]

        Nq, Nk = q_patches.size(1), kv_patches.size(1)

        q = self.q_proj(q_patches).view(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)  # [B, heads, Nq, d]
        k = self.k_proj(kv_patches).view(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_proj(kv_patches).view(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (C // self.num_heads) ** 0.5)
        attn = F.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, Nq, d]
        out = out.transpose(1, 2).reshape(B, Nq, C)  # [B, Nq, C]
        out = self.out_proj(out)

        Hq_p = rHq // self.patch_size
        Wq_p = rWq // self.patch_size
        out = out.transpose(1, 2).reshape(B, C, Hq_p, Wq_p)

        out = F.interpolate(out, size=(rHq, rWq), mode='bilinear', align_corners=False)
        return out


class STDiff(nn.Module):
    def __init__(self, in_channel, dim, num_heads=8, patch_size=4):
        super().__init__()
        self.short_cut_1 = nn.Conv2d(in_channel, dim, kernel_size=3, padding=1)
        self.short_cut_2 = nn.Conv2d(in_channel, dim, kernel_size=3, padding=1)
        self.caam = CAAM(channels=dim, expansion=4, drop=0.)
        self.statt = STAtt(dim, dim, num_heads=num_heads, patch_size=patch_size)
    
    def forward(self, xA, xB, xC):

        xA = self.short_cut_1(xA)
        xB = self.short_cut_2(xB)

        diff_res = torch.abs(xA - xB)

        xA, xB, c_weights = self.caam(xA, xB)

        diff = torch.abs(xA - xB)
        diff = self.statt(diff, xC) + diff + diff_res

        return diff, c_weights

if __name__ == '__main__':
    model = STDiff(in_channel=192, dim=128, num_heads=8, patch_size=4)
    xA = torch.randn(2, 192, 50, 50)
    xB = torch.randn(2, 192, 50, 50)
    xC = torch.randn(2, 12, 128, 40, 40)
    out, c_weights = model(xA, xB, xC)
    print(out.shape)
    print(c_weights.shape)