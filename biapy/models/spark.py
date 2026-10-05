"""
SparK: sparse and hierarchical masked image modeling for convolutional networks, adapted to BiaPy.

Reference: Tian et al., "Designing BERT for Convolutional Networks: Sparse and Hierarchical
Masked Modeling", ICLR 2023. Original code: https://github.com/keyu-tian/SparK
(pretrain/encoder.py, pretrain/decoder.py, pretrain/spark.py), Copyright (c) ByteDance, Inc.
and its affiliates.

Differences with the original implementation:

- The encoder is the encoder of BiaPy's U-NeXt V1 (``down_path``, ``downsample_layers`` and
  ``bottleneck``). Parameter names are kept, so the pretrained encoder weights can be loaded
  directly into a ``U_NeXt_V1`` model for fine-tuning (see ``SparK.encoder_state_dict``).
- ``ConvNeXtBlock_V1`` is turned into ``SparseConvNeXtBlock_V1``, which masks its residual
  branch before the addition. Without this, the biases of the block's Linear layers write
  non-zero values into masked positions, which the next depthwise convolution then reads.
  The original SparK ``SparseConvNeXtBlock`` does the same masking.
- Any number of input channels (1 for EM), rectangular inputs and a configurable decoder width.
- ``forward`` returns ``{"loss", "pred", "mask"}`` like BiaPy's MAE, so the model plugs into the
  ``masking`` pretext task of BiaPy's self-supervised workflow.

Only 2D data is supported.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import Permute

from biapy.models.blocks import ConvNeXtBlock_V1
from biapy.models.unext_v1 import U_NeXt_V1


# --------------------------------------------------------------------------------------------
# Sparse operators
# --------------------------------------------------------------------------------------------
# Mask of the current forward pass, shape (B, 1, f_h, f_w), dtype bool. True = visible (active).
# It is a module-level variable, as in the original SparK: every sparse layer reads it. This is
# safe with DistributedDataParallel (one process per GPU) but not with nn.DataParallel.
_cur_active: Optional[torch.Tensor] = None


def _get_active_ex_or_ii(H: int, W: int, returning_active_ex: bool = True):
    """Upsample the current mask to (H, W); return it, or the indices of its visible positions."""
    if _cur_active is None:
        raise RuntimeError("SparK: the mask is not set. Sparse layers can only run inside SparK.forward.")
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3)
    if returning_active_ex:
        return active_ex  # (B, 1, H, W)
    return active_ex.squeeze(1).nonzero(as_tuple=True)  # (bi, hi, wi)


class SparseConv2d(nn.Conv2d):
    """Dense convolution followed by zeroing the masked positions (submanifold sparse conv)."""

    def forward(self, x):
        x = super().forward(x)
        return x * _get_active_ex_or_ii(x.shape[2], x.shape[3])


class SparseMaxPooling(nn.MaxPool2d):
    def forward(self, x):
        x = super().forward(x)
        return x * _get_active_ex_or_ii(x.shape[2], x.shape[3])


class SparseAvgPooling(nn.AvgPool2d):
    def forward(self, x):
        x = super().forward(x)
        return x * _get_active_ex_or_ii(x.shape[2], x.shape[3])


def _sparse_bn(bn_forward, x):
    """Apply a BatchNorm1d-like forward only on the visible positions of a (B, C, H, W) tensor."""
    ii = _get_active_ex_or_ii(x.shape[2], x.shape[3], returning_active_ex=False)
    bhwc = x.permute(0, 2, 3, 1)
    nc = bn_forward(bhwc[ii])  # (N_visible, C): statistics only from visible positions
    out = torch.zeros(bhwc.shape, dtype=nc.dtype, device=x.device)
    out[ii] = nc
    return out.permute(0, 3, 1, 2)


class SparseBatchNorm2d(nn.BatchNorm1d):
    def forward(self, x):
        return _sparse_bn(super().forward, x)


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    def forward(self, x):
        return _sparse_bn(super().forward, x)


class SparseConvNeXtLayerNorm(nn.LayerNorm):
    """
    LayerNorm over channels that keeps masked positions at zero.

    ``data_format``: ``channels_last`` for (B, H, W, C) inputs, ``channels_first`` for (B, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", sparse=True):
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(data_format)
        super().__init__(normalized_shape, eps, elementwise_affine=True)
        self.data_format = data_format
        self.sparse = sparse

    def forward(self, x):
        if x.ndim != 4:
            raise NotImplementedError("SparseConvNeXtLayerNorm only supports 4D tensors (2D images)")
        if self.data_format == "channels_last":  # (B, H, W, C)
            if not self.sparse:
                return super().forward(x)
            ii = _get_active_ex_or_ii(x.shape[1], x.shape[2], returning_active_ex=False)
            nc = super().forward(x[ii])
            out = torch.zeros(x.shape, dtype=nc.dtype, device=x.device)
            out[ii] = nc
            return out
        # channels_first, (B, C, H, W)
        bhwc = x.permute(0, 2, 3, 1)
        if not self.sparse:
            return super().forward(bhwc).permute(0, 3, 1, 2)
        ii = _get_active_ex_or_ii(x.shape[2], x.shape[3], returning_active_ex=False)
        nc = super().forward(bhwc[ii])
        out = torch.zeros(bhwc.shape, dtype=nc.dtype, device=x.device)
        out[ii] = nc
        return out.permute(0, 3, 1, 2)

    def extra_repr(self):
        return super().extra_repr() + f", data_format={self.data_format}, sparse={self.sparse}"


class SparseConvNeXtBlock_V1(ConvNeXtBlock_V1):
    """
    ``ConvNeXtBlock_V1`` whose residual branch is masked before the residual addition.

    Instances are created by ``dense_model_to_sparse`` by changing the class of an existing
    ``ConvNeXtBlock_V1``, so parameter names (``block.*``, ``layer_scale``) do not change.
    """

    def forward(self, x):
        result = self.block(x)
        if self.layer_scale is not None:
            result = self.layer_scale * result
        result = result * _get_active_ex_or_ii(result.shape[2], result.shape[3])
        return x + self.stochastic_depth(result)


def dense_model_to_sparse(m: nn.Module, sbn: bool = False) -> nn.Module:
    """Recursively replace the layers of ``m`` by their sparse versions, copying the weights."""
    oup = m
    if isinstance(m, nn.Conv2d):
        bias = m.bias is not None
        oup = SparseConv2d(
            m.in_channels,
            m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=bias,
            padding_mode=m.padding_mode,
        )
        oup.weight.data.copy_(m.weight.data)
        if bias:
            oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, nn.MaxPool2d):
        oup = SparseMaxPooling(
            m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            return_indices=m.return_indices,
            ceil_mode=m.ceil_mode,
        )
    elif isinstance(m, nn.AvgPool2d):
        oup = SparseAvgPooling(
            m.kernel_size,
            m.stride,
            m.padding,
            ceil_mode=m.ceil_mode,
            count_include_pad=m.count_include_pad,
            divisor_override=m.divisor_override,
        )
    elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        cls = SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d
        oup = cls(
            m.num_features,
            eps=m.eps,
            momentum=m.momentum,
            affine=m.affine,
            track_running_stats=m.track_running_stats,
        )
        if m.affine:
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
        if m.track_running_stats:
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
    elif isinstance(m, nn.LayerNorm) and not isinstance(m, SparseConvNeXtLayerNorm):
        if len(m.normalized_shape) != 1 or not m.elementwise_affine:
            raise NotImplementedError("SparK: only LayerNorm over one channel dim with affine params is supported")
        oup = SparseConvNeXtLayerNorm(m.normalized_shape[0], eps=m.eps, data_format="channels_last", sparse=True)
        oup.weight.data.copy_(m.weight.data)
        oup.bias.data.copy_(m.bias.data)
    elif isinstance(m, ConvNeXtBlock_V1) and not isinstance(m, SparseConvNeXtBlock_V1):
        m.__class__ = SparseConvNeXtBlock_V1
        oup = m
    elif isinstance(m, (nn.Conv1d, nn.Conv3d, nn.ConvTranspose2d, nn.BatchNorm3d)):
        raise NotImplementedError(f"SparK: no sparse version of {type(m).__name__}")

    for name, child in m.named_children():
        oup.add_module(name, dense_model_to_sparse(child, sbn=sbn))
    return oup


# Leaf layers that are either sparse or act on each position independently.
_SPARK_ALLOWED_LEAVES = (
    SparseConv2d,
    SparseMaxPooling,
    SparseAvgPooling,
    SparseBatchNorm2d,
    SparseSyncBatchNorm2d,
    SparseConvNeXtLayerNorm,
    nn.Linear,
    nn.GELU,
    nn.ReLU,
    nn.Identity,
    nn.Dropout,
    Permute,
    StochasticDepth,
)
# Containers whose forward we know. Any other container could mix positions in its forward.
_SPARK_ALLOWED_CONTAINERS = (nn.Sequential, nn.ModuleList, SparseConvNeXtBlock_V1)


def _check_sparse_encoder(module: nn.Module, prefix: str):
    """Fail loudly if a layer without a sparse version is left in the encoder (it would leak)."""
    for name, m in module.named_modules(prefix=prefix):
        has_children = len(list(m.children())) > 0
        if has_children and not isinstance(m, _SPARK_ALLOWED_CONTAINERS):
            raise NotImplementedError(f"SparK: unsupported block '{name}' ({type(m).__name__}) in the encoder")
        if not has_children and not isinstance(m, _SPARK_ALLOWED_LEAVES):
            raise NotImplementedError(f"SparK: layer '{name}' ({type(m).__name__}) has no sparse version")


# --------------------------------------------------------------------------------------------
# Light U-Net decoder (dense)
# --------------------------------------------------------------------------------------------
class SparKUNetBlock(nn.Module):
    """U-Net decoder block with 2x upsampling."""

    def __init__(self, cin, cout, bn2d):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False),
            bn2d(cin),
            nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            bn2d(cout),
        )

    def forward(self, x):
        return self.conv(self.up_sample(x))


class SparKLightDecoder(nn.Module):
    """Decoder from SparK: log2(up_sample_ratio) UNet blocks, width halved at every block."""

    def __init__(self, up_sample_ratio: int, width: int = 768, out_chans: int = 3, sbn: bool = False):
        super().__init__()
        if up_sample_ratio < 2 or (up_sample_ratio & (up_sample_ratio - 1)) != 0:
            raise ValueError(f"SparK: the encoder downsampling ratio must be a power of 2 (got {up_sample_ratio})")
        self.width = width
        n = round(math.log2(up_sample_ratio))
        channels = [width // 2**i for i in range(n + 1)]
        if channels[-1] < 1:
            raise ValueError(f"SparK: decoder width {width} is too small for {n} upsampling blocks")
        bn2d = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec = nn.ModuleList(
            [SparKUNetBlock(cin, cout, bn2d) for (cin, cout) in zip(channels[:-1], channels[1:])]
        )
        self.proj = nn.Conv2d(channels[-1], out_chans, kernel_size=1, stride=1, bias=True)
        self._initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = d(x)
        return self.proj(x)

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # nn.ConvTranspose2d is not a subclass of nn.Conv2d
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def extra_repr(self):
        return f"width={self.width}"


# --------------------------------------------------------------------------------------------
# SparK
# --------------------------------------------------------------------------------------------
class SparK(nn.Module):
    """
    SparK pretraining model with a sparse U-NeXt V1 encoder.

    Parameters
    ----------
    image_shape : tuple
        ``(y, x, channels)``. y and x must be divisible by the encoder downsampling ratio
        (32 for U-NeXt V1 with 5 feature maps and ``stem_k_size=2``).
    encoder : str
        Only ``"unext_v1"`` for now.
    feature_maps, upsample_layer, z_down, cn_layers, layer_scale, stochastic_depth_prob, isotropy, stem_k_size
        Same meaning as in ``U_NeXt_V1``. They must match the model used for fine-tuning.
    mask_ratio : float
        Fraction of patches that are masked. One patch = one pixel of the deepest feature map.
    decoder_width : int
        Channels of the decoder at its deepest scale; halved at every upsampling.
    densify_norm : str
        Normalization applied to the sparse features before filling them: ``"ln"``, ``"bn"`` or ``"none"``.
    norm_pix_loss : bool
        Normalize each target patch with its own mean and std before the L2 loss.
    """

    ENCODER_PREFIXES = ("down_path.", "downsample_layers.", "bottleneck.")

    def __init__(
        self,
        image_shape=(256, 256, 1),
        encoder="unext_v1",
        feature_maps=[16, 32, 64, 128, 256],
        upsample_layer="convtranspose",
        z_down=[2, 2, 2, 2],
        cn_layers=[2, 2, 2, 2, 2],
        layer_scale=1e-6,
        stochastic_depth_prob=0.1,
        isotropy=True,
        stem_k_size=2,
        mask_ratio=0.6,
        decoder_width=256,
        densify_norm="ln",
        norm_pix_loss=True,
    ):
        super().__init__()
        if len(image_shape) != 3:
            raise ValueError("SparK in BiaPy only supports 2D data: image_shape must be (y, x, channels)")
        if encoder != "unext_v1":
            raise ValueError(f"SparK: encoder '{encoder}' not supported. Available: ['unext_v1']")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("SparK: mask_ratio must be in (0, 1)")

        H, W, C = image_shape
        self.in_chans = C
        self.ndim = 2
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # Build a dense U-NeXt V1 and keep only its encoder, converted to sparse layers.
        dense = U_NeXt_V1(
            image_shape=tuple(image_shape),
            feature_maps=list(feature_maps),
            upsample_layer=upsample_layer,
            z_down=list(z_down),
            output_channels=[1],
            stochastic_depth_prob=stochastic_depth_prob,
            layer_scale=layer_scale,
            cn_layers=list(cn_layers),
            isotropy=isotropy,
            stem_k_size=stem_k_size,
        )

        # Total downsampling ratio of the encoder = patch size of the mask
        stem_stride = dense.down_path[0][0].stride
        if stem_stride[0] != stem_stride[1]:
            raise ValueError("SparK: the stem must downsample y and x equally")
        for ds in dense.downsample_layers:
            if tuple(ds[-1].stride) != (2, 2):
                raise ValueError("SparK: every encoder downsampling must be 2x2")
        ratio = stem_stride[0] * 2 ** len(dense.downsample_layers)
        if H % ratio != 0 or W % ratio != 0:
            raise ValueError(f"SparK: DATA.PATCH_SIZE {image_shape[:2]} must be divisible by {ratio}")

        self.down_path = dense_model_to_sparse(dense.down_path)
        self.downsample_layers = dense_model_to_sparse(dense.downsample_layers)
        self.bottleneck = dense_model_to_sparse(dense.bottleneck)
        del dense  # the U-NeXt decoder and heads are not used
        _check_sparse_encoder(self.down_path, "down_path")
        _check_sparse_encoder(self.downsample_layers, "downsample_layers")
        _check_sparse_encoder(self.bottleneck, "bottleneck")

        self.downsample_ratio = ratio
        self.fmap_h, self.fmap_w = H // ratio, W // ratio
        n_patches = self.fmap_h * self.fmap_w
        self.len_keep = round(n_patches * (1 - mask_ratio))
        if not 1 <= self.len_keep < n_patches:
            raise ValueError(f"SparK: mask_ratio {mask_ratio} leaves {self.len_keep} of {n_patches} patches visible")

        # Feature maps returned by the encoder, from largest (stride ratio/16) to smallest (stride ratio)
        self.enc_feat_map_chs = list(feature_maps)
        self.hierarchy = len(self.enc_feat_map_chs)

        self.dense_decoder = SparKLightDecoder(ratio, width=decoder_width, out_chans=C, sbn=False)
        if self.hierarchy > len(self.dense_decoder.dec):
            raise ValueError("SparK: the encoder has more scales than the decoder has upsampling blocks")

        # Densify layers, from the smallest feature map (i=0, bottleneck) to the largest
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        d_width = decoder_width
        for i, e_width in enumerate(reversed(self.enc_feat_map_chs)):
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            nn.init.trunc_normal_(p, mean=0.0, std=0.02, a=-0.02, b=0.02)
            self.mask_tokens.append(p)

            if self.densify_norm_str == "bn":
                norm = SparseBatchNorm2d(e_width)
            elif self.densify_norm_str == "ln":
                norm = SparseConvNeXtLayerNorm(e_width, data_format="channels_first", sparse=True)
            elif self.densify_norm_str == "none":
                norm = nn.Identity()
            else:
                raise ValueError("SparK: densify_norm must be 'ln', 'bn' or 'none'")
            self.densify_norms.append(norm)

            if i == 0 and e_width == d_width:
                proj = nn.Identity()
            else:
                k = 1 if i == 0 else 3
                proj = nn.Conv2d(e_width, d_width, kernel_size=k, stride=1, padding=k // 2, bias=True)
            self.densify_projs.append(proj)
            d_width //= 2

    # ---------------------------------------------------------------------------------------
    def generate_mask(self, B: int, device) -> torch.Tensor:
        """Random mask of shape (B, 1, f_h, f_w), True = visible. Different for every sample."""
        n = self.fmap_h * self.fmap_w
        idx = torch.rand(B, n, device=device).argsort(dim=1)[:, : self.len_keep]
        active = torch.zeros(B, n, dtype=torch.bool, device=device)
        active.scatter_(dim=1, index=idx, value=True)
        return active.view(B, 1, self.fmap_h, self.fmap_w)

    def forward_features(self, x: torch.Tensor, active_b1ff: torch.Tensor) -> List[torch.Tensor]:
        """Sparse encoder. Returns the feature maps from the largest to the smallest."""
        global _cur_active
        _cur_active = active_b1ff
        r = self.downsample_ratio
        active_b1hw = active_b1ff.repeat_interleave(r, dim=2).repeat_interleave(r, dim=3)
        x = x * active_b1hw

        feats = []
        x = self.down_path[0](x)  # stem
        for down, pool in zip(self.down_path[1:], self.downsample_layers):
            x = down(x)
            feats.append(x)
            x = pool(x)
        feats.append(self.bottleneck(x))
        return feats

    def forward(self, x: torch.Tensor, active_b1ff: Optional[torch.Tensor] = None, return_just_preds: bool = False):
        """
        Returns
        -------
        dict
            - ``"loss"``: L2 loss on the masked patches only.
            - ``"pred"``: reconstruction as patches, shape (B, L, ratio*ratio*C). If ``norm_pix_loss``,
              it is de-normalized with the mean and std of each input patch (for visualization
              and metrics only, it carries no gradient).
            - ``"mask"``: shape (B, L), 1 = masked, 0 = visible (same convention as BiaPy's MAE).

            If ``return_just_preds`` is True, only the reconstructed image (B, C, H, W) is returned
            and the mask is fixed (seed 0), as in BiaPy's MAE (used for the BioImage Model Zoo sample).
        """
        if return_just_preds:
            torch.manual_seed(0)  # deterministic mask, as in BiaPy's MAE
        if active_b1ff is None:
            active_b1ff = self.generate_mask(x.shape[0], x.device)

        # Encode, then densify from the smallest feature map to the largest
        feats = self.forward_features(x, active_b1ff)
        feats.reverse()
        cur_active = active_b1ff
        to_dec = []
        for i, f in enumerate(feats):
            f = self.densify_norms[i](f)
            f = torch.where(cur_active.expand_as(f), f, self.mask_tokens[i].expand_as(f))
            to_dec.append(self.densify_projs[i](f))
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        # Decode and compute the loss on the masked patches
        rec = self.dense_decoder(to_dec)
        inp, rec = self.patchify(x), self.patchify(rec)
        if self.norm_pix_loss:
            mean = inp.mean(dim=-1, keepdim=True)
            std = (inp.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
            target = (inp - mean) / std
        else:
            mean, std, target = 0.0, 1.0, inp
        l2 = ((rec - target) ** 2).mean(dim=2)  # (B, L)
        mask = (~active_b1ff).flatten(1).to(l2.dtype)  # (B, L), 1 = masked
        loss = (l2 * mask).sum() / (mask.sum() + 1e-8)

        pred = (rec * std + mean).detach()
        if return_just_preds:
            return self.unpatchify(pred)
        return {"loss": loss, "pred": pred, "mask": mask}

    # ---------------------------------------------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, L, p*p*C)."""
        p, h, w = self.downsample_ratio, self.fmap_h, self.fmap_w
        B, C = imgs.shape[:2]
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum("bchpwq->bhwpqc", x)
        return x.reshape(B, h * w, p * p * C)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, p*p*C) -> (B, C, H, W)."""
        p, h, w = self.downsample_ratio, self.fmap_h, self.fmap_w
        B, C = x.shape[0], x.shape[-1] // (p * p)
        x = x.reshape(B, h, w, p, p, C)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(B, C, h * p, w * p)

    @torch.no_grad()
    def save_images(self, _x, _y, _mask, dtype):
        """
        Images for visualization, same interface as BiaPy's MAE.

        Returns reconstruction, masked input and input with the masked patches replaced by the
        reconstruction, each as a NumPy array of shape (N, H, W, C).
        """
        p = self.downsample_ratio
        rec = self.unpatchify(_y).float().cpu()
        mask = _mask.view(-1, 1, self.fmap_h, self.fmap_w).float()
        mask = mask.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3).cpu()  # 1 = masked
        x = _x.float().cpu()
        im_masked = x * (1 - mask)
        im_paste = x * (1 - mask) + rec * mask

        def to_np(t):
            return t.numpy().transpose(0, 2, 3, 1).astype(dtype)

        return to_np(rec), to_np(im_masked), to_np(im_paste)

    def encoder_state_dict(self) -> dict:
        """Encoder weights only, with the same keys as in ``U_NeXt_V1``."""
        return {k: v for k, v in self.state_dict().items() if k.startswith(self.ENCODER_PREFIXES)}
