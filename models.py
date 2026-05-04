# ─────────────────────────────────────────────
# CELL 1 — Imports
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'PyTorch version: {torch.__version__}')
print(f'timm version: {timm.__version__}')

# ─────────────────────────────────────────────
# CELL 2 — Novel Block: Asymmetry-Aware Cross-Attention Gate (ACAG)
#
# MATHEMATICAL DERIVATION:
#
# Given CNN feature map F ∈ ℝ^(B×C_cnn×H×W)
#   F_h = horizontal flip of F
#   F_v = vertical flip of F
#   A   = |F - F_h| + |F - F_v|     (asymmetry residual)
#
# A captures pixel-level deviation from symmetry axes.
# For benign nevi (symmetric): A ≈ 0 everywhere
# For melanoma (asymmetric):   A has large values at irregular regions
#
# Cross-Attention with asymmetry-biased query:
#   Q = W_q(F_cnn) + λ·pool(A)    ← asymmetry injects clinical prior
#   K = W_k(F_vit)
#   V = W_v(F_vit)
#   ACAG = softmax(QKᵀ / √d_k) · V
#
# This is cross-attention: CNN features (Q) attend to ViT tokens (K,V),
# guided by the asymmetry signal A.
# ─────────────────────────────────────────────

class ACACrossAttentionGate(nn.Module):
    """
    Asymmetry-Aware Cross-Attention Gate (ACAG)
    
    Merges:
      - cnn_feat : (B, C_cnn, H, W)  — local spatial features from EfficientNet
      - vit_feat : (B, N+1, C_vit)   — patch tokens from DeiT (N patches + CLS token)
    
    Returns:
      - fused : (B, d_model) — fused representation for classification
    """
    def __init__(self, cnn_dim: int, vit_dim: int, d_model: int = 512,
                 n_heads: int = 8, asym_lambda: float = 0.5):
        super().__init__()
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.d_k        = d_model // n_heads
        self.asym_lambda = asym_lambda
        
        # ── Projection: CNN features → Q space ──────
        # Flatten spatial dims (H*W) → treat each spatial location as a token
        self.proj_q = nn.Linear(cnn_dim, d_model)
        
        # ── Projection: ViT tokens → K,V space ──────
        self.proj_k = nn.Linear(vit_dim, d_model)
        self.proj_v = nn.Linear(vit_dim, d_model)
        
        # ── Asymmetry branch: compress asymmetry map → query bias ──
        # Input: C_cnn channels → output: d_model (same dim as Q)
        self.asym_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # B×C_cnn×1×1
            nn.Flatten(),              # B×C_cnn
            nn.Linear(cnn_dim, d_model),
            nn.ReLU(),
        )
        
        # ── Multi-head output projection ─────────────
        self.out_proj = nn.Linear(d_model, d_model)
        
        # ── Layer norm + dropout ─────────────────────
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        # ── Final pooling: (B, N_q, d_model) → (B, d_model) ─
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def compute_asymmetry_map(self, feat: torch.Tensor) -> torch.Tensor:
        """
        A = |F - F_h| + |F - F_v|
        feat: (B, C, H, W)
        Returns: (B, C, H, W)
        """
        F_h = torch.flip(feat, dims=[-1])   # horizontal flip
        F_v = torch.flip(feat, dims=[-2])   # vertical flip
        return (feat - F_h).abs() + (feat - F_v).abs()

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        """
        cnn_feat: (B, C_cnn, H, W)
        vit_feat: (B, N_tokens, C_vit)   — includes CLS token at index 0
        """
        B, C_cnn, H, W = cnn_feat.shape
        
        # ── Step 1: Compute asymmetry residual ────────
        A = self.compute_asymmetry_map(cnn_feat)   # B×C_cnn×H×W
        
        # ── Step 2: Project asymmetry → query bias ────
        asym_bias = self.asym_proj(A)               # B×d_model
        asym_bias = asym_bias.unsqueeze(1)           # B×1×d_model (broadcast across query tokens)
        
        # ── Step 3: CNN features → query tokens ───────
        # Flatten spatial: (B, C_cnn, H, W) → (B, H*W, C_cnn)
        cnn_tokens = cnn_feat.flatten(2).permute(0, 2, 1)   # B×(H*W)×C_cnn
        Q = self.proj_q(cnn_tokens)        # B×(H*W)×d_model
        Q = Q + self.asym_lambda * asym_bias  # Inject asymmetry prior into queries
        
        # ── Step 4: ViT tokens → key/value ───────────
        K = self.proj_k(vit_feat)          # B×N_tokens×d_model
        V = self.proj_v(vit_feat)          # B×N_tokens×d_model
        
        # ── Step 5: Multi-head scaled dot-product attention ─
        # Reshape for multi-head: (B, heads, seq, d_k)
        def split_heads(x):   # x: (B, N, d_model)
            return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        Q_h = split_heads(Q)   # B×heads×(H*W)×d_k
        K_h = split_heads(K)   # B×heads×N_tokens×d_k
        V_h = split_heads(V)   # B×heads×N_tokens×d_k
        
        # Scaled dot-product: QKᵀ / √d_k
        scale  = self.d_k ** -0.5
        attn   = torch.matmul(Q_h, K_h.transpose(-2, -1)) * scale   # B×heads×(H*W)×N_tokens
        attn   = F.softmax(attn, dim=-1)                              # row-stochastic matrix
        attn   = self.dropout(attn)
        
        # Aggregate values
        out    = torch.matmul(attn, V_h)           # B×heads×(H*W)×d_k
        out    = out.transpose(1, 2).contiguous()  # B×(H*W)×heads×d_k
        out    = out.view(B, H*W, self.d_model)    # B×(H*W)×d_model
        
        # ── Step 6: Output projection + norm ────────
        out    = self.out_proj(out)        # B×(H*W)×d_model
        out    = self.norm(out)
        
        # ── Step 7: Pool to single vector ───────────
        # (B, H*W, d_model) → (B, d_model, H*W) → avg pool → (B, d_model)
        fused  = self.global_pool(out.transpose(1, 2)).squeeze(-1)
        
        return fused, attn   # return attn for visualization in Notebook 07

print('✅ ACAG Block defined')
print('   Key novelty: Asymmetry residual |F - F_h| + |F - F_v| biases the attention queries')
print('   Clinical basis: Asymmetry is ABCDE criterion "A" for melanoma detection')

# ─────────────────────────────────────────────
# CELL 3 — FiLM (Feature-wise Linear Modulation) Layer
#
# MATHEMATICAL DERIVATION:
#   Given metadata vector m ∈ ℝ^(META_DIM):
#     γ, β = split(MLP(m))   ← two learned affine parameters
#     FiLM(x) = γ ⊙ x + β   ← element-wise scale + shift
#
# Geometrically: FiLM applies a metadata-conditioned affine
# transformation to the feature manifold.
# Example: for a 75-year-old male with head lesion (high-risk),
# γ will upscale the malignant-direction axes of the feature space.
# ─────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM)
    Conditions visual feature x on metadata vector m.
    
    Reference: Perez et al., 2018. "FiLM: Visual Reasoning with a General
    Conditioning Layer." AAAI-2018.
    """
    def __init__(self, meta_dim: int, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.meta_dim = meta_dim
        self.feat_dim = feat_dim
        
        # MLP produces 2 × feat_dim outputs: [γ | β]
        self.film_net = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2 * feat_dim),
        )
        
        # Initialize γ→1, β→0 so FiLM starts as identity (stable training)
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.constant_(self.film_net[-1].bias[:feat_dim], 1.0)   # γ init = 1
        nn.init.zeros_(self.film_net[-1].bias[feat_dim:])            # β init = 0

    def forward(self, x: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, feat_dim) — visual features
        metadata: (B, meta_dim) — tabular metadata vector
        Returns:  (B, feat_dim) — modulated features
        """
        params  = self.film_net(metadata)         # B × (2*feat_dim)
        gamma   = params[:, :self.feat_dim]       # B × feat_dim  (scale)
        beta    = params[:, self.feat_dim:]       # B × feat_dim  (shift)
        return gamma * x + beta

print('✅ FiLM Layer defined')
print('   γ init=1, β init=0 → training starts from identity (stable gradient flow)')

# ─────────────────────────────────────────────
# CELL 4 — DermoViT: Full Dual-Stream Model
# ─────────────────────────────────────────────

class DermoViT(nn.Module):
    """
    DermoViT: Dual-Stream Dermoscopy Vision Transformer
    
    Architecture:
      Stream 1 (Local): EfficientNet-B2 CNN — captures borders, texture, pigment
      Stream 2 (Global): DeiT-Small          — captures global shape, symmetry, long-range context
      Fusion:            ACAG Block          — asymmetry-guided cross-attention
      Conditioning:      FiLM Layer          — metadata-conditioned affine transformation
      Head:              2-layer MLP         — binary classification (Malignant/Benign)
    """
    
    def __init__(self,
                 meta_dim:       int   = 30,      # FiLM input dimension (from NB04)
                 d_model:        int   = 512,     # Internal fusion dimension
                 n_heads:        int   = 8,
                 asym_lambda:    float = 0.5,
                 drop_rate:      float = 0.3,
                 freeze_backbone:bool  = True):   # Freeze for warmup phase
        super().__init__()
        
        # ════════════════════════════════════════════
        # STREAM 1: EfficientNet-B2 (Local CNN)
        # WHY B2 not B4: B2 runs on 8GB GPU; B4 needs 16GB+
        # Feature map size: (B, 1408, 7, 7) at input 224×224
        # ════════════════════════════════════════════
        self.cnn_backbone = timm.create_model(
            'efficientnet_b2',
            pretrained=True,
            features_only=True,      # return intermediate feature maps
            out_indices=[4],          # only last stage
        )
        cnn_dim = self.cnn_backbone.feature_info[4]['num_chs']  # 1408
        
        if freeze_backbone:
            for p in self.cnn_backbone.parameters():
                p.requires_grad = False
        
        # ════════════════════════════════════════════
        # STREAM 2: DeiT-Small (Global ViT)
        # WHY DeiT not ViT: DeiT uses distillation token for
        # data-efficient training — doesn't need JFT-300M pre-training
        # Output: (B, 197, 384) for 224×224 input (196 patches + 1 CLS)
        # ════════════════════════════════════════════
        self.vit_backbone = timm.create_model(
            'deit_small_patch16_224',
            pretrained=True,
            num_classes=0,            # remove classification head, get (B, 384) CLS token
        )
        vit_dim = self.vit_backbone.embed_dim  # 384 for DeiT-Small
        
        if freeze_backbone:
            for p in self.vit_backbone.parameters():
                p.requires_grad = False
        
        # ════════════════════════════════════════════
        # FUSION: ACAG Block (Novel Custom Block)
        # ════════════════════════════════════════════
        self.acag = ACACrossAttentionGate(
            cnn_dim=cnn_dim,
            vit_dim=vit_dim,
            d_model=d_model,
            n_heads=n_heads,
            asym_lambda=asym_lambda,
        )
        
        # ════════════════════════════════════════════
        # CONDITIONING: FiLM Layer
        # ════════════════════════════════════════════
        self.film = FiLMLayer(meta_dim=meta_dim, feat_dim=d_model)
        
        # ════════════════════════════════════════════
        # CLASSIFICATION HEAD
        # ════════════════════════════════════════════
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 1),      # Binary: logit for Malignant
        )
        
        self._cnn_dim = cnn_dim
        self._vit_dim = vit_dim
        self._d_model = d_model

    def forward(self, image: torch.Tensor, metadata: torch.Tensor):
        """
        image:    (B, 3, 224, 224)
        metadata: (B, meta_dim)
        Returns:  logit (B, 1), acag_attn (for visualization)
        """
        # ── CNN Stream ────────────────────────────
        cnn_feats = self.cnn_backbone(image)[0]  # (B, 1408, 7, 7)
        
        # ── ViT Stream ────────────────────────────
        # forward_features returns (B, N+1, 384) — all tokens
        vit_feats = self.vit_backbone.forward_features(image)  # (B, 197, 384)
        
        # ── ACAG Fusion ───────────────────────────
        fused, acag_attn = self.acag(cnn_feats, vit_feats)  # (B, d_model)
        
        # ── FiLM Conditioning ─────────────────────
        fused = self.film(fused, metadata)         # (B, d_model)
        
        # ── Classification Head ───────────────────
        logit = self.head(fused)                   # (B, 1)
        
        return logit, acag_attn

    def unfreeze_backbone(self):
        """Call after warmup epochs to enable end-to-end fine-tuning."""
        for p in self.cnn_backbone.parameters():
            p.requires_grad = True
        for p in self.vit_backbone.parameters():
            p.requires_grad = True
        print('Backbone unfrozen — full end-to-end training active')

print('✅ DermoViT model class defined')

# ─────────────────────────────────────────────
# CELL 6 — Architecture Visualization Diagram
# ─────────────────────────────────────────────
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#0F0F1A')
fig.patch.set_facecolor('#0F0F1A')

def draw_box(ax, x, y, w, h, text, color, text_color='white', fontsize=9):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0.1', facecolor=color, edgecolor='white',
        linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold',
            wrap=True, multialignment='center')

def arrow(ax, x1, y1, x2, y2, color='white'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Input
draw_box(ax, 6.5, 8.5, 3, 1, '🖼️ Dermoscopy Image\n(B × 3 × 224 × 224)', '#2C3E50')

# Two streams
draw_box(ax, 0.5, 5.8, 4.5, 2, 'LOCAL STREAM\nEfficientNet-B2 CNN\n(B × 1408 × 7 × 7)\nTexture  Borders  Color', '#1A6B8A')
draw_box(ax, 11, 5.8, 4.5, 2, 'GLOBAL STREAM\nDeiT-Small ViT\n(B × 197 × 384)\nShape  Symmetry  Context', '#6B4B8A')

# ACAG
draw_box(ax, 4.5, 3.2, 7, 1.8,
    '⚡ NOVEL: ACAG Block (Asymmetry-Aware Cross-Attention Gate)\n'
    'A = |F - Fₕ| + |F - Fᵥ|   →   Q = W_q(CNN) + λ·A\n'
    'ACAG = softmax(QKᵀ / √d_k) · V_vit    [d_model=512]',
    '#B7410E', fontsize=8.5)

# Metadata + FiLM
draw_box(ax, 0.5, 2.8, 3, 1.5,
    '📊 Metadata\nage, sex, site\ntbp_lv_symm,\ntbp_lv_norm_border\n(B × META_DIM)', '#1E6E1E', fontsize=7.5)
draw_box(ax, 4.5, 1.5, 7, 1.2,
    '🎛️ FiLM Layer: γ(m)⊙x + β(m)   [affine modulation by metadata]', '#8A6B1A')

# Head
draw_box(ax, 5.5, 0.2, 5, 0.9, '🎯 Classification Head → Malignant Logit', '#7B241C')

# Arrows
arrow(ax, 8, 8.5, 4.5, 7.8)   # img → cnn
arrow(ax, 8, 8.5, 11.5, 7.8)  # img → vit
arrow(ax, 4.5, 6.8, 6, 5.0)   # cnn → acag
arrow(ax, 13, 5.8, 10.5, 5.0) # vit → acag
arrow(ax, 3.5, 2.8, 6, 3.4)   # meta → acag (via FiLM)
arrow(ax, 8, 3.2, 8, 2.7)     # acag → film
arrow(ax, 8, 1.5, 8, 1.1)     # film → head

ax.set_title('DermoViT Architecture — Dual-Stream with Novel ACAG Fusion Block',
             color='white', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
import os
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/05_dermovit_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='#0F0F1A')
plt.show()

