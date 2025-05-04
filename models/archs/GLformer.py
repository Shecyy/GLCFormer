import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from models.archs import arch_util


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, window_count=None):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.window_count = window_count
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        if window_count is not None:
            self.local_weight = nn.Parameter(torch.ones(size=[window_count, window_count]), requires_grad=True)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def global_forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    def local_forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        win_h = h // self.window_count
        win_w = w // self.window_count
        out = None

        for i in range(self.window_count):
            win_outs = []
            for j in range(self.window_count):
                start_x, end_x = i * win_h, (i + 1) * win_h
                start_y, end_y = j * win_w, (j + 1) * win_w
                curr_win_h, curr_win_w = win_h, win_w

                if i == self.window_count - 1:
                    end_x = x.shape[2]
                    curr_win_h = end_x - start_x
                if j == self.window_count - 1:
                    end_y = x.shape[3]
                    curr_win_w = end_y - start_y

                win_q = q[:, :, start_x:end_x, start_y:end_y]
                win_k = k[:, :, start_x:end_x, start_y:end_y]
                win_v = v[:, :, start_x:end_x, start_y:end_y]

                win_q = rearrange(win_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                win_k = rearrange(win_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
                win_v = rearrange(win_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

                win_q = torch.nn.functional.normalize(win_q, dim=-1)
                win_k = torch.nn.functional.normalize(win_k, dim=-1)

                attn = (win_q @ win_k.transpose(-2, -1)) * self.temperature
                attn = attn.softmax(dim=-1)

                win_out = (attn @ win_v)
                win_out = rearrange(win_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=curr_win_h,
                                    w=curr_win_w)

                win_outs.append(win_out * torch.tanh(self.local_weight[i, j]))
                pass
            win_outs = torch.cat(win_outs, dim=3)
            if out is None:
                out = win_outs
            else:
                out = torch.cat([out, win_outs], dim=2)
            pass
        out = self.project_out(out)
        return out

    def forward(self, x, is_local=False):
        if is_local:
            out = self.local_forward(x)
        else:
            out = self.global_forward(x)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, window_count=4):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        self.global_attn = Attention(dim, num_heads, bias)
        self.local_attn = Attention(dim, num_heads, bias, window_count=window_count)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = arch_util.FeedForward4(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x, light_diff = x
        light_diff = F.interpolate(light_diff, size=x.shape[2:])

        norm1_x = self.norm1(x)
        local_attention = self.local_attn(norm1_x, is_local=True)
        global_attention = self.global_attn(norm1_x, is_local=False)

        x = x + (1 - light_diff) * global_attention + light_diff * local_attention
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Expert(nn.Module):

    def __init__(self, in_c=48, embed_dim=48, bias=False, win_count=4):
        super(Expert, self).__init__()

        self.win_count = win_count
        self.expert = nn.Sequential(
            nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, groups=in_c, bias=bias),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True))
        self.merge = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=bias)
        pass

    def forward(self, x):
        x = self.expert(x)
        pool = F.adaptive_avg_pool2d(x, [self.win_count, self.win_count])
        pool = F.interpolate(pool, size=x.shape[2:])
        x = self.merge(torch.cat([x, pool], dim=1))
        return x


class ExpertEmbed(nn.Module):

    def __init__(self, in_c=48, embed_dim=48, bias=False, window_count=4):
        super(ExpertEmbed, self).__init__()
        self.window_count = window_count

        self.experts = nn.ModuleList()
        for win_count in range(self.window_count, 1, -1):
            self.experts.append(Expert(in_c, embed_dim, bias, win_count))
        pass

    def forward(self, x, light):
        residual = x
        up_near = nn.UpsamplingNearest2d(size=light.shape[2:])
        for i, win_count in enumerate(range(self.window_count, 1, -1)):
            win_h, win_w = light.shape[2] // win_count, light.shape[3] // win_count
            pool = F.avg_pool2d(light, [win_h, win_w])
            indicator = up_near(pool)  # 亮度引导器
            x = self.experts[i](x) * (1 - indicator)
            pass
        return residual + x

    pass


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- GLformer -----------------------
class GLformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 1, 1, 1],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=1,
                 bias=True,
                 LayerNorm_type='WithBias',
                 opt=None):
        super(GLformer, self).__init__()

        window_count = opt['network_G']['window_count']

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.expert_embed = ExpertEmbed(dim, dim, window_count=window_count)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type, window_count=window_count) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, window_count=window_count) for i in
            range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, window_count=window_count) for i in
            range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, window_count=window_count) for i in
            range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, window_count=window_count) for i in
            range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, window_count=window_count) for i in
            range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, window_count=window_count) for i in
            range(num_blocks[0])])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        pass

    def forward(self, inp_img, light=None, light_diff=None):
        # encoder
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1 = self.expert_embed(inp_enc_level1, light)

        out_enc_level1 = self.encoder_level1((inp_enc_level1, light_diff))

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2((inp_enc_level2, light_diff))

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3((inp_enc_level3, light_diff))

        # bottleneck
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent((inp_enc_level4, light_diff))

        # decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3((inp_dec_level3, light_diff))

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2((inp_dec_level2, light_diff))

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1((inp_dec_level1, light_diff))

        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1
