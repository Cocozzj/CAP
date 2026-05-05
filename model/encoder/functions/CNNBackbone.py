# ===========================================================================
# CNNBackbone
# ===========================================================================
import torch
import torch.nn as nn
import timm
from transformers import VideoMAEModel

def center_crop_to_multiple(x, mult):
    # x: [B,T,C,H,W]
    B, T, C, H, W = x.shape
    Hc = (H // mult) * mult
    Wc = (W // mult) * mult
    top = (H - Hc) // 2
    left = (W - Wc) // 2
    if Hc != H or Wc != W:
        x = x[..., top:top+Hc, left:left+Wc]
    return x, (Hc, Wc, top, left)

class ResNetBackbone(nn.Module):
    """
    ResNet34 feature extractor for clips shaped [Batch, T, C, H, W].

    Returns:
        feats: [Batch, T, C_out, Hf, Wf]
        (H, W, top, left)
    """
    def __init__(self, pretrained=True, out_index=3, chunk=128, normalize=True,center_crop_to_patch=True,patch_size=16):
        super().__init__()
        self.net = timm.create_model(
            "resnet34",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        with torch.no_grad():
            feats = self.net(torch.zeros(1, 3, 224, 224))
        self._chosen = int(out_index if out_index is not None else 3)
        self.out_channels = feats[self._chosen].shape[1]
        
        self.chunk = int(chunk)
        self.normalize = bool(normalize)
        
        self.center_crop_to_patch = bool(center_crop_to_patch)
        stride_map = {0: 2, 1: 4, 2: 8, 3: 16, 4: 32}
        default_mult = stride_map.get(self._chosen, 16)
        self.patch_size = default_mult if patch_size is None else int(patch_size)
        if self.normalize:
            # channel-first: [1,1,1,3,1,1] will broadcast to [B,T,C,H,W]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

    def forward(self, clip: torch.Tensor):
        # clip: [Batch, T, C, H, W]
        assert clip.ndim == 5, f"Expected [Batch,T,C,H,W], got {clip.shape}"
        B, T, C, H, W = clip.shape
        assert C == 3, f"Expected C=3, got {C}"
        top, left = 0, 0
        is_uint8 = clip.dtype == torch.uint8

        if self.center_crop_to_patch:
            clip, (H, W, top, left) = center_crop_to_multiple(clip, self.patch_size)

        # Flatten time → [B*T, C, H, W], keep original dtype to save memory
        x = clip.reshape(B * T, C, H, W)

        # Normalization buffers reshaped for 4-D [1, 3, 1, 1]
        mean4d = self.mean.view(1, 3, 1, 1) if self.normalize else None
        std4d = self.std.view(1, 3, 1, 1) if self.normalize else None

        feats_out = []
        for i in range(0, x.size(0), self.chunk):
            chunk = x[i:i + self.chunk].float()
            if is_uint8:
                chunk /= 255.0
            if self.normalize:
                chunk = (chunk - mean4d) / std4d
            fmap = self.net(chunk)[self._chosen]  # [chunk, C_out, Hf, Wf]
            feats_out.append(fmap)

        feats_out = torch.cat(feats_out, dim=0).contiguous()    # [B*T, C_out, Hf, Wf]
        _, C_out, Hf, Wf = feats_out.shape
        feats_out = feats_out.view(B, T, C_out, Hf, Wf).contiguous()         # [B, T, C_out, Hf, Wf]
        return feats_out, (H, W, top, left)
    
class VideoMAEBackbone(nn.Module):
    """
    VideoMAE feature extractor for clips shaped [Batch, T, C, H, W].

    Returns:
        feats: [Batch, Ttok, C_out, Htok, Wtok]
        (H, W, top, left)
    """
    def __init__(
        self,
        variant: str = "base",          # "tiny" | "small" | "base" | "large"
        tubelet_size: int =1,
        proj_dim: int = 512,
        chunk: int = 32,
        normalize: bool = True,
        center_crop_to_patch: bool = True,
        patch_size: int = 16
    ):
        super().__init__()
        model_name = {
            "tiny":  "MCG-NJU/videomae-tiny",
            "small": "MCG-NJU/videomae-small",
            "base":  "MCG-NJU/videomae-base",
            "large": "MCG-NJU/videomae-large",
        }.get(variant, "MCG-NJU/videomae-base")

        self.model = VideoMAEModel.from_pretrained(model_name)
        cfg = self.model.config
        self.hidden_size = int(cfg.hidden_size)
        ps = getattr(cfg, "patch_size", patch_size)
        if isinstance(ps, (list, tuple)):
            ps = ps[-1] if len(ps) > 1 else ps[0]
        self.patch_size = int(ps)
        self.tubelet_size = int(getattr(cfg, "tubelet_size", tubelet_size))

        self.proj = nn.Conv2d(self.hidden_size, proj_dim, kernel_size=1)
        self.out_channels = int(proj_dim)

        self.chunk = int(chunk)
        self.normalize = bool(normalize)
        self.center_crop_to_patch = bool(center_crop_to_patch)

        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)  # [1,1,C,1,1]
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

    def _align_time_to_tubelet(self, x: torch.Tensor):
        # x: [B,T,C,H,W]
        B, T, C, H, W = x.shape
        tz = self.tubelet_size
        if tz <= 1:
            return x, T
        if T % tz == 0:
            return x, T

        # Prefer center-crop to largest multiple; if too short, pad.
        T_target = (T // tz) * tz
        if T_target <= 0:
            pad = tz - T
            pad_tail = torch.zeros(B, pad, C, H, W, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tail], dim=1)
            return x, x.shape[1]
        start = (T - T_target) // 2
        x = x[:, start:start + T_target]
        return x, T_target

    def forward(self, clip: torch.Tensor):
        # clip: [Batch, T, C, H, W]
        assert clip.ndim == 5, f"Expected [Batch,T,C,H,W], got {clip.shape}"
        B, T, C, H, W = clip.shape
        assert C == 3
        top, left = 0, 0

        x = (clip.float() / 255.0) if clip.dtype == torch.uint8 else clip.float()
        if self.normalize:
            x = (x - self.mean) / self.std

        if self.center_crop_to_patch:
            x, (H, W, top, left) = center_crop_to_multiple(x, self.patch_size)

        x, T = self._align_time_to_tubelet(x)
        
        tz = self.tubelet_size
        Htok, Wtok = H // self.patch_size, W // self.patch_size
        Ttok = T // tz

        outs = []
        for i in range(0, B, self.chunk):
            batch = x[i:i+self.chunk].permute(0, 2, 1, 3, 4).contiguous()  # [bs, C, T, H, W]
            out = self.model(pixel_values=batch)
            tokens = out.last_hidden_state  # [bs, N, hidden]
            expected = Ttok * Htok * Wtok
            if tokens.size(1) == expected + 1:
                tokens = tokens[:, 1:, :]
            elif tokens.size(1) != expected:
                raise RuntimeError(
                    f"Token count mismatch: got {tokens.size(1)} tokens, expected {expected} or {expected+1}"
                )
            # [bs, N, hidden] -> [bs, Ttok, Htok, Wtok, hidden]
            tokens = tokens.contiguous().view(batch.size(0), Ttok, Htok, Wtok, self.hidden_size)
            t2d = tokens.permute(0, 1, 4, 2, 3).contiguous()          # [bs, Ttok, hidden, Htok, Wtok]
            t2d = t2d.view(-1, self.hidden_size, Htok, Wtok).contiguous()  # [bs*Ttok, hidden, Htok, Wtok]
            t2d = self.proj(t2d)                                      # [bs*Ttok, C_out, Htok, Wtok]
            t2d = t2d.view(batch.size(0), Ttok, self.out_channels, Htok, Wtok).contiguous()
            outs.append(t2d)

        feats = torch.cat(outs, dim=0).contiguous()  # [B, Ttok, C_out, Htok, Wtok]
        return feats, (H, W, top, left)
    
class ViTBackbone(nn.Module):
    """
    2D ViT feature extractor for clips shaped [Batch, T, C, H, W].

    Returns:
        feats: [Batch, T, C_out, Htok, Wtok]
        (H, W, top, left)
    """
    def __init__(
        self,
        variant: str = "base",  # or "vit_base_patch16_224", "deit_small_patch16_224", etc.
        pretrained: bool = True,
        proj_dim: int = 512,
        chunk: int = 128,
        normalize: bool = True,
        center_crop_to_patch: bool = True,
        patch_size: int = 16
    ):
        super().__init__()
        model_name = {
            "small": "vit_small_patch16_224",
            "base":  "vit_base_patch16_224",
        }.get(variant, "vit_base_patch16_224")
        
        self.normalize = bool(normalize)
        self.center_crop_to_patch = bool(center_crop_to_patch)
        self.chunk = int(chunk)

        # Create ViT without classifier head.
        # global_pool='' keeps token outputs behavior consistent across timm ViT/DeiT variants.
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # Patch size (assume square)
        ps = getattr(self.vit, "patch_embed", None)
        if ps is None or not hasattr(ps, "patch_size"):
            raise ValueError(f"{model_name} does not look like a timm ViT/DeiT with patch_embed.patch_size.")
        patch = ps.patch_size
        self.patch_size = int(patch[0] if isinstance(patch, (tuple, list)) else patch)

        # Hidden dim
        embed_dim = getattr(self.vit, "embed_dim", None)
        if embed_dim is None:
            # some models store this differently
            embed_dim = getattr(self.vit, "num_features", None)
        self.hidden_size = int(embed_dim)

        # Project token embedding dim -> desired channel dim
        self.proj = nn.Conv2d(self.hidden_size, proj_dim, kernel_size=1)
        self.out_channels = int(proj_dim)

        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)
    
    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor, Htok: int, Wtok: int):
        # tokens: [N, Npatch, hidden]
        N, Npatch, D = tokens.shape
        assert Npatch == Htok * Wtok, f"Npatch={Npatch} != Htok*Wtok={Htok*Wtok}"
        x = tokens.view(N, Htok, Wtok, D).permute(0, 3, 1, 2).contiguous()  # [N, D, Htok, Wtok]
        return x

    def forward(self, clip: torch.Tensor):
        # clip: [Batch, T, C, H, W]
        assert clip.ndim == 5, f"Expected [Batch,T,C,H,W], got {clip.shape}"
        B, T, C, H, W = clip.shape
        assert C == 3, f"Expected C=3, got {C}"
        top, left = 0, 0

        x = (clip.float() / 255.0) if clip.dtype == torch.uint8 else clip.float()
        if self.normalize:
            x = (x - self.mean) / self.std

        if self.center_crop_to_patch:
            x, (H, W, top, left) = center_crop_to_multiple(x, self.patch_size)

        Htok, Wtok = H // self.patch_size, W // self.patch_size

        # Flatten time: [B*T, C, H, W]
        x = x.reshape(B * T, C, H, W).contiguous()

        feats_out = []
        for i in range(0, x.size(0), self.chunk):
            xi = x[i:i + self.chunk]  # [n, C, H, W]

            # timm ViT forward_features usually returns token sequence [n, 1+Npatch, D] (with CLS)
            tok = self.vit.forward_features(xi)

            # Some timm models may return a single vector if global_pool is set; we forced global_pool=""
            # So expect tokens.
            if tok.ndim != 3:
                raise RuntimeError(f"Unexpected forward_features output shape: {tok.shape}. "
                                   f"Make sure global_pool='' and num_classes=0.")

            # Drop CLS token if present
            # Most ViT/DeiT: [n, 1+Npatch, D]
            if tok.size(1) == Htok * Wtok + 1:
                tok = tok[:, 1:, :]                # drop CLS
            elif tok.size(1) == Htok * Wtok:
                pass                               # no CLS token
            else:
                raise RuntimeError(
                    f"Token count mismatch: got {tok.size(1)}, "
                    f"expected {Htok*Wtok} or {Htok*Wtok+1}"
                )

            # [n, Npatch, D] -> [n, D, Htok, Wtok] -> proj -> [n, C_out, Htok, Wtok]
            fmap = self._tokens_to_map(tok, Htok, Wtok)
            fmap = self.proj(fmap)
            feats_out.append(fmap)

        feats_out = torch.cat(feats_out, dim=0).contiguous()  # [B*T, C_out, Htok, Wtok]
        feats_out = feats_out.view(B, T, self.out_channels, Htok, Wtok).contiguous()
        return feats_out, (H, W, top, left)

class CNNBackbone(nn.Module):
    """
    Input:  frames [B, V, T, C, H, W]
    Output: feats  [B, V, Tp, C_out, Hf, Wf]
            (Hcrop, Wcrop, top, left)
    """
    def __init__(self, backbone_type: str, resnet_param: dict, vit_param: dict, videomae_param: dict):
        super().__init__()
        self.backbone_type = backbone_type.lower()
        if self.backbone_type == "videomae":
            self.backbone = VideoMAEBackbone(**videomae_param)
        elif self.backbone_type == "vit":
            self.backbone = ViTBackbone(**vit_param)
        elif self.backbone_type == "resnet34":
            self.backbone = ResNetBackbone(**resnet_param)
        else:
            raise ValueError(
                f"Unknown backbone_type={backbone_type}. Use 'resnet34' | 'vit' | 'videomae'."
            )

        self.out_channels = self.backbone.out_channels

    def forward(self, frames: torch.Tensor):
        B, V, T, C, H, W = frames.shape
        frames = frames.view(B * V, T, C, H, W).contiguous()
        feats, crop_info = self.backbone(frames)
        # crop_info = (Hcrop, Wcrop, top, left) — needed for pixel-space mapping
        BV, Tp, C_out, Hf, Wf = feats.shape
        feats = feats.view(B, V, Tp, C_out, Hf, Wf).contiguous()
        return feats, crop_info
    


