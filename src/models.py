"""Models for the photodamage retrain.

ResNet50MTL  — ImageNet-pretrained ResNet50 trunk, two 3-way FC heads (photodamage main +
               pigmentation auxiliary), joint loss = CE_photo + alpha * CE_pigment. Mirrors
               Kahler et al. BJD 2026 § "CNN model development".

DermLIPMTL   — CLIP-style ViT-B/16 from Derm1M / DermLIP (Siyuan Yan et al.) as the image
               encoder, with the same two 3-way heads. Supports two training modes:
                 mode="linear-probe"  freezes the backbone (train heads only).
                 mode="full"          unfreezes everything (allows backbone fine-tuning).

DualStreamMTL — wrapper for paired colour + grayscale tile inputs. Supports five fusion
                strategies (selected with `fusion=`):
                  "concat"         — two independent shared-backbone forwards, concatenate
                                     CLS embeddings → 1024-d, classify. (baseline)
                  "cross-attn"     — two independent forwards expose CLS + patch tokens;
                                     2 cross-attention blocks where CLS_c queries patch_g
                                     and CLS_g queries patch_c (CrossViT-style).
                  "two-cls"        — single shared forward over [CLS_c, CLS_g,
                                     patches_c, patches_g] with modality embeddings;
                                     concatenate the two CLS outputs → 1024-d.
                  "modality-embed" — single shared forward over [CLS, patches_c, patches_g]
                                     with modality embeddings on the patches; one CLS → 512-d.
                  "lora"           — per-modality MLP adapters inserted after each transformer
                                     block (CLS-token-only); concatenate CLS → 1024-d.

References supporting the fusion choices: see `training/notes/dual_stream_fusion_research.md`.
"""
from typing import Optional, Tuple

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _make_head(in_dim: int, out_dim: int, head_dropout: float = 0.0) -> nn.Module:
    """One classification head — `Linear` if no dropout, else `Dropout → Linear`."""
    if head_dropout > 0.0:
        return nn.Sequential(nn.Dropout(head_dropout), nn.Linear(in_dim, out_dim))
    return nn.Linear(in_dim, out_dim)


class ResNet50MTL(nn.Module):
    """Shared ResNet50 trunk with two 3-way classification heads."""

    def __init__(self, n_photodamage: int = 3, n_pigmentation: int = 3,
                 pretrained: bool = True, head_dropout: float = 0.0):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head_photo = _make_head(self.feat_dim, n_photodamage, head_dropout)
        self.head_pigment = _make_head(self.feat_dim, n_pigmentation, head_dropout)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.head_photo(feat), self.head_pigment(feat)


class DermLIPMTL(nn.Module):
    """DermLIP (ViT-B/16) image encoder + two 3-way heads.

    `mode`:
      - "linear-probe": backbone is frozen; only the heads update.
      - "full":         backbone is trainable end-to-end.

    `use_pre_projection`: if True, return the 768-d pre-projection CLS feature
        (the protocol Radford et al. 2021 Appendix A actually used for CLIP linear
        probing — the post-projection 512-d embedding bias-collapses task-irrelevant
        directions which we may need for non-semantic targets like photodamage severity).

    `head_dropout`: probability of dropout applied before each classification head
        (0.0 = no dropout; 0.3 is a standard regularisation default).
    """

    def __init__(self, n_photodamage: int = 3, n_pigmentation: int = 3,
                 mode: str = "linear-probe",
                 checkpoint: str = "hf-hub:redlessone/DermLIP_ViT-B-16",
                 use_pre_projection: bool = False,
                 head_dropout: float = 0.0):
        super().__init__()
        if mode not in ("linear-probe", "full"):
            raise ValueError(f"mode must be 'linear-probe' or 'full' (got {mode!r})")
        self.mode = mode
        self.use_pre_projection = use_pre_projection
        clip_model, _, _ = open_clip.create_model_and_transforms(checkpoint)
        self.backbone = clip_model.visual
        # 768 if pre-projection, 512 if post-projection.
        self.feat_dim = 768 if use_pre_projection else 512

        self.head_photo = _make_head(self.feat_dim, n_photodamage, head_dropout)
        self.head_pigment = _make_head(self.feat_dim, n_pigmentation, head_dropout)

        if mode == "linear-probe":
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward x through the visual tower; return either 512-d (post-proj) or 768-d (pre-proj)."""
        v = self.backbone
        if self.use_pre_projection:
            # Match open_clip VisionTransformer.forward exactly, stopping before visual.proj.
            # Do not manually transpose tokens: DermLIP's transformer is batch_first=True.
            x = v._embeds(x)
            x = v.transformer(x)
            pooled, _ = v._pool(x)
            return pooled  # (B, 768)
        return v(x)  # (B, 512), standard open_clip output

    def forward(self, x: torch.Tensor):
        if self.mode == "linear-probe":
            with torch.no_grad():
                feat = self._encode(x)
        else:
            feat = self._encode(x)
        return self.head_photo(feat), self.head_pigment(feat)


# ---------------------------------------------------------------------------
# Helpers to manually drive DermLIP's ViT-B/16 visual tower forward
# ---------------------------------------------------------------------------

def _dermlip_embed_patches(visual, x: torch.Tensor) -> torch.Tensor:
    """Patch embedding + add CLS + add positional embedding. Returns (B, 197, 768)."""
    return visual._embeds(x)


def _dermlip_run_transformer(visual, tokens: torch.Tensor) -> torch.Tensor:
    """Run transformer + ln_post, preserving the full sequence including CLS.

    DermLIP's open_clip VisionTransformer is batch_first=True, so `tokens` must stay
    in (B, N, D) format. Do not call visual._pool() here: for pool_type='tok' it
    returns patch tokens with the CLS removed, which breaks CLS-based fusion paths.
    """
    tokens = visual.transformer(tokens)
    return visual.ln_post(tokens)


def _dermlip_proj(visual, tokens_768: torch.Tensor) -> torch.Tensor:
    """Project from 768-d to 512-d via visual.proj (matches CLIP image-text alignment).
    `tokens_768` may be (B, 768) for a single CLS or (B, N, 768) for a batch of tokens."""
    if visual.proj is None:
        return tokens_768
    return tokens_768 @ visual.proj


# ---------------------------------------------------------------------------
# Cross-attention fusion block (option b)
# ---------------------------------------------------------------------------

class CrossAttnBlock(nn.Module):
    """One CrossViT-style block: CLS from one stream queries patch tokens from the other.

    Operates entirely in 512-d (post-proj space) so the pretrained image-text alignment
    is preserved. Two `MultiheadAttention` ops per block (one per direction)."""

    def __init__(self, dim: int = 512, n_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(dim)
        self.norm_kv1 = nn.LayerNorm(dim)
        self.attn1 = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_q2 = nn.LayerNorm(dim)
        self.norm_kv2 = nn.LayerNorm(dim)
        self.attn2 = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        h = int(dim * mlp_ratio)
        self.norm_mlp1 = nn.LayerNorm(dim)
        self.mlp1 = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))
        self.norm_mlp2 = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, cls_c, tok_c, cls_g, tok_g):
        # cls_c, cls_g: (B, 1, 512). tok_c, tok_g: (B, 196, 512).
        d1, _ = self.attn1(self.norm_q1(cls_c), self.norm_kv1(tok_g), self.norm_kv1(tok_g))
        cls_c = cls_c + d1
        d2, _ = self.attn2(self.norm_q2(cls_g), self.norm_kv2(tok_c), self.norm_kv2(tok_c))
        cls_g = cls_g + d2
        cls_c = cls_c + self.mlp1(self.norm_mlp1(cls_c))
        cls_g = cls_g + self.mlp2(self.norm_mlp2(cls_g))
        return cls_c, cls_g


# ---------------------------------------------------------------------------
# Per-modality MLP adapter (option d — "LoRA")
# ---------------------------------------------------------------------------

class ModalityAdapter(nn.Module):
    """Bottleneck MLP residual: dim -> rank -> dim. Inserted after the backbone forward;
    applied only to the CLS token. Parameter-efficient, ~50K params at default rank."""

    def __init__(self, dim: int = 512, rank: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(x)))


# ---------------------------------------------------------------------------
# DualStreamMTL with 5 fusion strategies
# ---------------------------------------------------------------------------

class DualStreamMTL(nn.Module):
    """Two-stream colour + grayscale wrapper for ResNet50 or DermLIP backbones.

    `fusion` ∈ {"concat", "cross-attn", "two-cls", "modality-embed", "lora"}.
    Non-DermLIP backbones only support "concat" (the other strategies rely on access to
    ViT patch tokens, which the ResNet50 wrapper doesn't expose)."""

    def __init__(self, inner: nn.Module,
                 fusion: str = "concat",
                 n_photodamage: int = 3, n_pigmentation: int = 3,
                 n_cross_blocks: int = 2,
                 head_dropout: float = 0.0):
        super().__init__()
        if fusion not in ("concat", "cross-attn", "two-cls", "modality-embed", "lora"):
            raise ValueError(f"unknown fusion {fusion!r}")
        self.inner = inner
        self.fusion = fusion
        self.is_dermlip = hasattr(inner, "mode")  # DermLIPMTL has `.mode`

        if not self.is_dermlip and fusion != "concat":
            raise ValueError(f"fusion={fusion!r} requires the DermLIP backbone; "
                             f"got {type(inner).__name__}")

        feat_dim = inner.feat_dim  # 512 for DermLIP, 2048 for ResNet50
        if fusion == "modality-embed":
            # Single CLS → only one feat_dim output
            self.feat_dim = feat_dim
        else:
            # All other strategies output 2× concatenated CLS
            self.feat_dim = 2 * feat_dim

        # Fresh heads on the new feature width.
        self.head_photo = _make_head(self.feat_dim, n_photodamage, head_dropout)
        self.head_pigment = _make_head(self.feat_dim, n_pigmentation, head_dropout)
        # Expose for the optimizer-group code in train.py
        inner.head_photo = self.head_photo
        inner.head_pigment = self.head_pigment

        # Per-fusion submodules.
        if self.is_dermlip:
            self.dim_768 = 768  # ViT-B/16 internal dim
            if fusion == "cross-attn":
                self.cross_blocks = nn.ModuleList([
                    CrossAttnBlock(dim=feat_dim) for _ in range(n_cross_blocks)
                ])
            elif fusion in ("two-cls", "modality-embed"):
                # Learnable modality embeddings, added to the 768-d patch tokens
                self.mod_embed_color = nn.Parameter(torch.zeros(1, 1, self.dim_768))
                self.mod_embed_gray = nn.Parameter(torch.zeros(1, 1, self.dim_768))
                nn.init.trunc_normal_(self.mod_embed_color, std=0.02)
                nn.init.trunc_normal_(self.mod_embed_gray, std=0.02)
                if fusion == "two-cls":
                    # A separate learnable CLS for the gray stream (the colour stream
                    # reuses the pretrained class_embedding).
                    self.cls_gray = nn.Parameter(torch.zeros(1, 1, self.dim_768))
                    nn.init.trunc_normal_(self.cls_gray, std=0.02)
            elif fusion == "lora":
                self.adapter_color = ModalityAdapter(dim=feat_dim, rank=64)
                self.adapter_gray = ModalityAdapter(dim=feat_dim, rank=64)

    # -------- backbone-call helpers --------

    def _bb_frozen(self):
        return self.is_dermlip and self.inner.mode == "linear-probe"

    def _forward_dermlip_get_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward x through DermLIP visual. Returns (cls_512, patch_tokens_512)."""
        v = self.inner.backbone
        ctx = torch.no_grad() if self._bb_frozen() else torch.enable_grad()
        with ctx:
            tokens = _dermlip_embed_patches(v, x)             # (B, 197, 768)
            tokens = _dermlip_run_transformer(v, tokens)      # (B, 197, 768)
            tokens = _dermlip_proj(v, tokens)                 # (B, 197, 512)
        cls = tokens[:, 0]                                    # (B, 512)
        patch = tokens[:, 1:]                                 # (B, 196, 512)
        return cls, patch

    def _forward_dermlip_cls_only(self, x: torch.Tensor) -> torch.Tensor:
        """Single-stream forward for DermLIP. Respects the inner model's use_pre_projection
        flag so DualStream + pre-projection works consistently."""
        if self._bb_frozen():
            with torch.no_grad():
                return self.inner._encode(x)
        return self.inner._encode(x)

    # -------- fusion-specific paths --------

    def _fwd_concat(self, x_color, x_gray):
        if not self.is_dermlip:
            # ResNet50 path: backbone returns the 2048-d pooled feature directly.
            f_c = self.inner.backbone(x_color)
            f_g = self.inner.backbone(x_gray)
        else:
            f_c = self._forward_dermlip_cls_only(x_color)
            f_g = self._forward_dermlip_cls_only(x_gray)
        return torch.cat([f_c, f_g], dim=-1)

    def _fwd_cross_attn(self, x_color, x_gray):
        cls_c, tok_c = self._forward_dermlip_get_tokens(x_color)
        cls_g, tok_g = self._forward_dermlip_get_tokens(x_gray)
        cls_c = cls_c.unsqueeze(1)  # (B, 1, 512)
        cls_g = cls_g.unsqueeze(1)
        for blk in self.cross_blocks:
            cls_c, cls_g = blk(cls_c, tok_c, cls_g, tok_g)
        return torch.cat([cls_c.squeeze(1), cls_g.squeeze(1)], dim=-1)

    def _shared_transformer_forward(self, tokens: torch.Tensor) -> torch.Tensor:
        v = self.inner.backbone
        tokens = v.transformer(tokens)
        return v.ln_post(tokens)

    def _embed_with_modality(self, x: torch.Tensor, mod_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (CLS_token_with_pos, patches_with_pos_and_modembed). Both un-projected (768-d).
        CLS gets the pretrained CLS pos embed; patches get the patch pos embed + modality embed."""
        v = self.inner.backbone
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)      # (B, 196, 768)
        pe = v.positional_embedding.to(x.dtype)                          # (197, 768)
        cls_pos = pe[0:1]                                                # (1, 768)
        patch_pos = pe[1:]                                               # (196, 768)
        cls_tok = v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1) + cls_pos
        patches = x + patch_pos + mod_embed                              # (B, 196, 768)
        return cls_tok, patches

    def _fwd_two_cls(self, x_color, x_gray):
        v = self.inner.backbone
        # Build colour stream tokens (no separate gray-CLS yet).
        cls_c, patches_c = self._embed_with_modality(x_color, self.mod_embed_color)
        cls_g_partial, patches_g = self._embed_with_modality(x_gray, self.mod_embed_gray)
        # cls_g_partial is the colour-stream CLS template; replace its embedding base with
        # the learnable gray CLS, then keep the CLS positional embedding it already has.
        cls_pos = v.positional_embedding.to(cls_g_partial.dtype)[0:1]
        cls_g = self.cls_gray.to(cls_g_partial.dtype).expand(cls_g_partial.shape[0], 1, -1) + cls_pos
        # Single transformer over [CLS_c, CLS_g, patches_c (196), patches_g (196)] = 394 tokens.
        tokens = torch.cat([cls_c, cls_g, patches_c, patches_g], dim=1)
        ctx = torch.no_grad() if self._bb_frozen() else torch.enable_grad()
        with ctx:
            tokens = v.patch_dropout(tokens)
            tokens = v.ln_pre(tokens)
            tokens = self._shared_transformer_forward(tokens)            # (B, 394, 768)
            tokens = _dermlip_proj(v, tokens)                             # (B, 394, 512)
        return torch.cat([tokens[:, 0], tokens[:, 1]], dim=-1)            # (B, 1024)

    def _fwd_modality_embed(self, x_color, x_gray):
        v = self.inner.backbone
        cls_c, patches_c = self._embed_with_modality(x_color, self.mod_embed_color)
        _, patches_g = self._embed_with_modality(x_gray, self.mod_embed_gray)
        # Single CLS, both sets of patches.
        tokens = torch.cat([cls_c, patches_c, patches_g], dim=1)          # (B, 393, 768)
        ctx = torch.no_grad() if self._bb_frozen() else torch.enable_grad()
        with ctx:
            tokens = v.patch_dropout(tokens)
            tokens = v.ln_pre(tokens)
            tokens = self._shared_transformer_forward(tokens)
            tokens = _dermlip_proj(v, tokens)                             # (B, 393, 512)
        return tokens[:, 0]                                               # (B, 512)

    def _fwd_lora(self, x_color, x_gray):
        f_c = self.adapter_color(self._forward_dermlip_cls_only(x_color))
        f_g = self.adapter_gray(self._forward_dermlip_cls_only(x_gray))
        return torch.cat([f_c, f_g], dim=-1)

    def forward(self, x_color: torch.Tensor, x_gray: torch.Tensor):
        if self.fusion == "concat":
            feat = self._fwd_concat(x_color, x_gray)
        elif self.fusion == "cross-attn":
            feat = self._fwd_cross_attn(x_color, x_gray)
        elif self.fusion == "two-cls":
            feat = self._fwd_two_cls(x_color, x_gray)
        elif self.fusion == "modality-embed":
            feat = self._fwd_modality_embed(x_color, x_gray)
        elif self.fusion == "lora":
            feat = self._fwd_lora(x_color, x_gray)
        return self.head_photo(feat), self.head_pigment(feat)


def build_model(backbone: str, mode: str = "full",
                dual_stream: bool = False, fusion: str = "concat",
                use_pre_projection: bool = False,
                loss_type: str = "ce",
                head_dropout: float = 0.0,
                **kwargs) -> nn.Module:
    """Factory.

    backbone:           'resnet50' or 'dermlip'.
    mode:               dermlip only — 'full' or 'linear-probe'.
    dual_stream:        if True, wrap the single-stream model in DualStreamMTL.
    fusion:             dual-stream fusion strategy. See DualStreamMTL docstring.
    use_pre_projection: dermlip only — use 768-d pre-projection CLS instead of 512-d post-projection.
    loss_type:          'ce' → heads output K logits; 'ordinal' → heads output K-1 logits
                        (CORN-style; pair with OrdinalMTLLoss + ordinal_logits_to_probs).
    head_dropout:       p of dropout before each classification head (default 0.0 = off).
    """
    # Translate the class count for the heads. Labels stay in [0, K-1] either way.
    n_photo = kwargs.get("n_photodamage", 3)
    n_pigment = kwargs.get("n_pigmentation", 3)
    if loss_type == "ordinal":
        kwargs = {**kwargs,
                  "n_photodamage": n_photo - 1,
                  "n_pigmentation": n_pigment - 1}
    elif loss_type != "ce":
        raise ValueError(f"loss_type must be 'ce' or 'ordinal' (got {loss_type!r})")

    # Inner model carries the dropout when dual_stream=False; otherwise the wrapper does.
    inner_kwargs = {**kwargs, "head_dropout": head_dropout if not dual_stream else 0.0}

    if backbone == "resnet50":
        inner = ResNet50MTL(**inner_kwargs)
    elif backbone == "dermlip":
        inner = DermLIPMTL(mode=mode, use_pre_projection=use_pre_projection, **inner_kwargs)
    else:
        raise ValueError(f"unknown backbone {backbone!r}")

    if not dual_stream:
        return inner
    return DualStreamMTL(inner, fusion=fusion,
                         n_photodamage=kwargs.get("n_photodamage", 3),
                         n_pigmentation=kwargs.get("n_pigmentation", 3),
                         head_dropout=head_dropout)


class MTLLoss(nn.Module):
    """Joint cross-entropy: L = CE_photo + alpha * CE_pigment.

    `label_smoothing` (PyTorch ≥ 1.10) softens the one-hot targets — distributes
    `eps` mass uniformly across all classes, which regularises against overconfident
    predictions and is the cheapest standard fix for train↔val gap overfit.

    `forward()` also accepts soft (B, K) targets for mixup; cross-entropy is then
    computed as -(soft_target * log_softmax(logits)).sum(-1).mean()."""

    def __init__(self, alpha: float = 1.0,
                 photo_weight: Optional[torch.Tensor] = None,
                 pigment_weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.ce_photo = nn.CrossEntropyLoss(weight=photo_weight, label_smoothing=label_smoothing)
        self.ce_pigment = nn.CrossEntropyLoss(weight=pigment_weight, label_smoothing=label_smoothing)

    @staticmethod
    def _soft_ce(logits, soft_targets):
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    def forward(self, logits_photo, logits_pigment, y_photo, y_pigment):
        if y_photo.dim() == 2:    # soft targets from mixup
            l1 = self._soft_ce(logits_photo, y_photo)
            l2 = self._soft_ce(logits_pigment, y_pigment)
        else:
            l1 = self.ce_photo(logits_photo, y_photo)
            l2 = self.ce_pigment(logits_pigment, y_pigment)
        return l1 + self.alpha * l2, l1.detach(), l2.detach()


class OrdinalMTLLoss(nn.Module):
    """CORN-style ordinal regression loss for K-class ordinal targets.

    Each head outputs (K-1) logits. For target y in {0, ..., K-1}, binary targets are
    t_k = 1{y > k} for k = 0, ..., K-2. Loss = BCE summed across the K-1 outputs.

    Reference: Cao et al., 'Rank consistent ordinal regression for neural networks
    with application to age estimation' (2020). Slightly looser than CORAL — does not
    constrain monotonicity — but in practice the network usually converges to monotone."""

    def __init__(self, alpha: float = 1.0,
                 n_classes_photo: int = 3, n_classes_pigment: int = 3):
        super().__init__()
        self.alpha = alpha
        self.K_photo = n_classes_photo
        self.K_pigment = n_classes_pigment
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def _build_targets(y: torch.Tensor, K: int) -> torch.Tensor:
        thresholds = torch.arange(K - 1, device=y.device).unsqueeze(0)   # (1, K-1)
        return (y.unsqueeze(1) > thresholds).float()                       # (B, K-1)

    def forward(self, logits_photo, logits_pigment, y_photo, y_pigment):
        t1 = self._build_targets(y_photo, self.K_photo)
        t2 = self._build_targets(y_pigment, self.K_pigment)
        l1 = self.bce(logits_photo, t1)
        l2 = self.bce(logits_pigment, t2)
        return l1 + self.alpha * l2, l1.detach(), l2.detach()


def mixup_batch(x, y_photo, y_pigment, alpha: float, n_classes: int = 3):
    """Mixup on a single-stream tensor or a (color, gray) dual-stream tuple.

    Samples lambda ~ Beta(alpha, alpha) once per BATCH, takes a permutation of the
    batch, and returns mixed inputs + soft (one-hot mixed) targets so MTLLoss can
    consume them via its soft-target path.

    Returns the same input shape (tensor or tuple) plus soft photo and pigment targets."""
    if alpha <= 0.0:
        return x, y_photo, y_pigment
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    if isinstance(x, (list, tuple)):
        bs = x[0].shape[0]
        device = x[0].device
        perm = torch.randperm(bs, device=device)
        x_mixed = tuple(lam * t + (1.0 - lam) * t[perm] for t in x)
    else:
        bs = x.shape[0]
        device = x.device
        perm = torch.randperm(bs, device=device)
        x_mixed = lam * x + (1.0 - lam) * x[perm]
    y_photo_oh = torch.nn.functional.one_hot(y_photo, n_classes).float()
    y_pigment_oh = torch.nn.functional.one_hot(y_pigment, n_classes).float()
    y_photo_mixed = lam * y_photo_oh + (1.0 - lam) * y_photo_oh[perm]
    y_pigment_mixed = lam * y_pigment_oh + (1.0 - lam) * y_pigment_oh[perm]
    return x_mixed, y_photo_mixed, y_pigment_mixed


def ordinal_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN ordinal logits (B, K-1) → class-probability vector (B, K).

    P(y = 0)    = 1 - σ(z_0)
    P(y = k)    = σ(z_{k-1}) - σ(z_k)        for k = 1, ..., K-2
    P(y = K-1)  = σ(z_{K-2})

    The differences are clipped to ≥ 0 (in case logits are non-monotonic) and the
    resulting vector is renormalised to sum to 1, so the output is a valid distribution."""
    sig = torch.sigmoid(logits)                       # (B, K-1) — P(y > k)
    B, Km1 = sig.shape
    K = Km1 + 1
    probs = torch.zeros(B, K, device=logits.device, dtype=logits.dtype)
    probs[:, 0] = 1.0 - sig[:, 0]
    for k in range(1, K - 1):
        probs[:, k] = sig[:, k - 1] - sig[:, k]
    probs[:, K - 1] = sig[:, K - 2]
    probs = probs.clamp_min(0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return probs
