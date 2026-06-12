"""
Titan Base Language Model — v0.3 "Architect"
=============================================
A decoder-only Transformer built from scratch in PyTorch.

Architecture upgrades from v0.2:
    RMSNorm
        Replaces LayerNorm on every block (pre-norm position unchanged).
        Faster, no mean-centering bias, used in LLaMA 2/3, Mistral, Falcon.
        No .bias parameter — existing checkpoints MUST be re-upscaled.

    SwiGLU activation (gate + up + down projection)
        Replaces the GELU (fc1 / fc2) MLP.
        Better gradient flow, sharper sparsity, used in PaLM, LLaMA, Mistral.
        Intermediate hidden dim is computed as (2/3 * d_ff) rounded to multiple
        of 64 to keep parameter count equivalent to the previous GELU MLP.
        State dict keys: gate_proj, up_proj, down_proj (was fc1, fc2).

    Grouped Query Attention (GQA)
        Q has n_heads heads; K and V share n_kv_heads heads (n_kv_heads <= n_heads).
        n_heads must be divisible by n_kv_heads.
        Set n_kv_heads == n_heads in the config for standard MHA (default).
        GQA reduces KV-cache memory by (n_heads / n_kv_heads) × at inference.
        State dict keys: q_proj, k_proj, v_proj, out_proj (was qkv_proj, out_proj).
        FlashAttention-2 handles GQA natively via its split q,k,v API.

    Gradient checkpointing
        Enabled via config.use_gradient_checkpointing = True.
        Trades compute for VRAM — essential for 1B+ on A100 40GB.
        Uses torch.utils.checkpoint.checkpoint per block.

Preserved from v0.2 (unchanged):
    Rotary Positional Embeddings (RoPE) — position-free token embeddings.
    FlashAttention-2 path with PyTorch SDPA and manual fallback.
    Pre-LayerNorm (now Pre-RMSNorm) block structure.
    Scaled residual projection initialization.
    Weight tying (token_embedding ↔ lm_head).

CHECKPOINT COMPATIBILITY:
    v0.3 state dict keys differ from v0.2 in three ways:
        1. *.ln1/ln2/ln_final keys: no longer have a .bias entry (RMSNorm).
        2. *.attn.qkv_proj → *.attn.q_proj + *.attn.k_proj + *.attn.v_proj
        3. *.mlp.fc1 / fc2 → *.mlp.gate_proj / up_proj / down_proj
    Use scripts/upscale_to_1b.py or scripts/upscale_to_3b.py to convert.
"""

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as gradient_checkpoint
from dataclasses import dataclass, field


# ─── FlashAttention-2 availability ───────────────────────────────────────────

def _check_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            print("[TitanLM] FlashAttention-2 : ACTIVE")
            return True
        print("[TitanLM] FlashAttention-2 : INACTIVE (no CUDA — using SDPA)")
        return False
    except ImportError:
        print("[TitanLM] FlashAttention-2 : INACTIVE (not installed — using SDPA)")
        return False

FLASH_ATTN_AVAILABLE = _check_flash_attn()


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class TitanConfig:
    """All model hyperparameters. Load from a YAML config via from_dict()."""
    vocab_size: int = 32000
    d_model: int   = 256
    n_heads: int   = 4
    n_kv_heads: int = 4      # GQA: set equal to n_heads for standard MHA
    n_layers: int  = 4
    d_ff: int      = 1024    # Target expanded dim; SwiGLU hidden = (2/3 * d_ff) ÷ 64 * 64
    max_seq_len: int = 2048
    dropout: float   = 0.1
    rope_base: float = 10000.0  # RoPE frequency base — increase for longer context
    tie_embeddings: bool = True
    use_gradient_checkpointing: bool = False

    # Computed at post_init — do not set manually
    d_head: int = field(init=False)
    swiglu_hidden: int = field(init=False)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        self.d_head = self.d_model // self.n_heads
        # SwiGLU hidden: equal param count to GELU MLP with d_ff
        # GELU: 2 × d_model × d_ff  | SwiGLU: 3 × d_model × h  → h = (2/3) × d_ff
        raw = int(2 * self.d_ff / 3)
        self.swiglu_hidden = (raw + 63) // 64 * 64  # round up to nearest 64

    @classmethod
    def from_dict(cls, cfg: dict) -> "TitanConfig":
        m = cfg["model"]
        n_heads = m["n_heads"]
        # n_kv_heads defaults to n_heads (standard MHA) if not specified
        n_kv_heads = m.get("n_kv_heads", n_heads)
        return cls(
            vocab_size=m["vocab_size"],
            d_model=m["d_model"],
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=m["n_layers"],
            d_ff=m["d_ff"],
            max_seq_len=m.get("max_seq_len", 2048),
            dropout=m.get("dropout", 0.1),
            rope_base=m.get("rope_base", 10000.0),
            tie_embeddings=m.get("tie_embeddings", True),
            use_gradient_checkpointing=m.get("gradient_checkpointing", False),
        )

    def to_dict(self) -> dict:
        return dict(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            rope_base=self.rope_base,
            tie_embeddings=self.tie_embeddings,
            gradient_checkpointing=self.use_gradient_checkpointing,
        )

    @property
    def n_kv_groups(self) -> int:
        return self.n_heads // self.n_kv_heads


# ─── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Faster than LayerNorm (no mean subtraction), no bias parameter.
    Used in LLaMA 2/3, Mistral 7B, Falcon, and most modern LLMs.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self._norm_shape = (d_model,)  # stored for F.rms_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.rms_norm: fused CUDA kernel (PyTorch 2.4+), numerically stable, bf16-safe
        return F.rms_norm(x, self._norm_shape, self.weight, self.eps)


# ─── Rotary Positional Embeddings (RoPE) ──────────────────────────────────────

def precompute_rope_freqs(
    d_head: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cosine/sine tables.
    Returns: cos_table (max_seq_len, d_head), sin_table (max_seq_len, d_head)
    """
    assert d_head % 2 == 0, f"d_head must be even for RoPE, got {d_head}"
    theta = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    pos   = torch.arange(max_seq_len).float()
    freqs = torch.outer(pos, theta)             # (max_seq_len, d_head/2)
    freqs = torch.cat([freqs, freqs], dim=-1)   # (max_seq_len, d_head)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K. Inputs: (B, n_heads, T, d_head)."""
    cos = cos[:q.shape[2]].unsqueeze(0).unsqueeze(0)  # (1,1,T,d_head)
    sin = sin[:q.shape[2]].unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ─── Grouped Query Attention ──────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with:
        - Grouped Query Attention (GQA) — Q has n_heads, K/V have n_kv_heads
        - Rotary Positional Embeddings (RoPE)
        - FlashAttention-2 > PyTorch SDPA > manual fallback
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups   = config.n_kv_groups
        self.d_head     = config.d_head
        self.d_model    = config.d_model
        self.dropout_p  = config.dropout

        # Separate Q / K / V projections (GQA-compatible)
        self.q_proj   = nn.Linear(config.d_model, config.n_heads    * config.d_head, bias=False)
        self.k_proj   = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.v_proj   = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_drop  = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # RoPE buffers (non-persistent — recomputed on device change)
        cos_t, sin_t = precompute_rope_freqs(config.d_head, config.max_seq_len, config.rope_base)
        self.register_buffer("rope_cos", cos_t, persistent=False)
        self.register_buffer("rope_sin", sin_t, persistent=False)

        # Causal mask (manual fallback only)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
              .view(1, 1, config.max_seq_len, config.max_seq_len),
            persistent=False,
        )

        self._use_flash = FLASH_ATTN_AVAILABLE
        self._use_sdpa  = (not self._use_flash) and hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project Q / K / V
        q = self.q_proj(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)  # (B,Hq,T,d)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)  # (B,Hk,T,d)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)  # (B,Hk,T,d)

        # Apply RoPE to Q and K
        q, k = apply_rope(q, k, self.rope_cos, self.rope_sin)

        # ── Attention path ────────────────────────────────────────────────────
        if self._use_flash:
            from flash_attn import flash_attn_func
            # flash_attn_func accepts (B, T, Hq, d) for q and (B, T, Hk, d) for k/v
            # and handles GQA natively when Hk < Hq
            q_fa = q.transpose(1, 2)   # (B, T, Hq, d_head)
            k_fa = k.transpose(1, 2)   # (B, T, Hk, d_head)
            v_fa = v.transpose(1, 2)   # (B, T, Hk, d_head)
            # Explicit bf16 cast — required when torch.compile/inductor is active
            q_fa = q_fa.to(torch.bfloat16)
            k_fa = k_fa.to(torch.bfloat16)
            v_fa = v_fa.to(torch.bfloat16)
            out  = flash_attn_func(q_fa, k_fa, v_fa, causal=True)  # (B, T, Hq, d_head)
            out  = out.reshape(B, T, C).to(x.dtype)  # cast back: flash_attn always returns bf16

        elif self._use_sdpa:
            # Expand K/V from n_kv_heads to n_heads for SDPA
            if self.n_groups > 1:
                k = k.repeat_interleave(self.n_groups, dim=1)  # (B, Hq, T, d_head)
                v = v.repeat_interleave(self.n_groups, dim=1)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, C)

        else:
            # Manual fallback — expand K/V for GQA
            if self.n_groups > 1:
                k = k.repeat_interleave(self.n_groups, dim=1)
                v = v.repeat_interleave(self.n_groups, dim=1)
            scale = math.sqrt(self.d_head)
            scores = (q @ k.transpose(-2, -1)) / scale           # (B, Hq, T, T)
            scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_drop(weights)
            out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_drop(self.out_proj(out))


# ─── SwiGLU MLP ───────────────────────────────────────────────────────────────

class TitanMLP(nn.Module):
    """
    SwiGLU feed-forward network.
    hidden_dim = (2/3 × d_ff) rounded to multiple of 64 to preserve parameter
    count vs. the previous GELU (fc1 + fc2) MLP.
    gate_proj, up_proj → hidden | down_proj → d_model
    Activation: SiLU(gate) * up  (the "SwiGLU" gate mechanism)
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        h = config.swiglu_hidden
        self.gate_proj = nn.Linear(config.d_model, h, bias=False)
        self.up_proj   = nn.Linear(config.d_model, h, bias=False)
        self.down_proj = nn.Linear(h, config.d_model, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ─── Transformer Block ────────────────────────────────────────────────────────

class TitanBlock(nn.Module):
    """
    Transformer decoder block.
    Pre-RMSNorm structure (unchanged from v0.2 Pre-LayerNorm position).
    Supports gradient checkpointing via config.use_gradient_checkpointing.
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn  = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp   = TitanMLP(config)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)


# ─── Full Model ───────────────────────────────────────────────────────────────

class TitanLM(nn.Module):
    """
    Titan Language Model v0.3 — decoder-only Transformer.
    RMSNorm + SwiGLU + GQA + RoPE + FlashAttention-2.
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.config = config
        self._use_grad_ckpt = config.use_gradient_checkpointing

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout   = nn.Dropout(config.dropout)
        self.blocks          = nn.ModuleList([TitanBlock(config) for _ in range(config.n_layers)])
        self.ln_final        = RMSNorm(config.d_model)
        self.lm_head         = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Scale residual projections for training stability
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[TitanLM v0.3] Parameters : {n_params:,}  (~{n_params/1e9:.3f}B)")
        print(f"[TitanLM v0.3] Arch        : d_model={config.d_model} | n_heads={config.n_heads} "
              f"| n_kv_heads={config.n_kv_heads} | n_layers={config.n_layers}")
        print(f"[TitanLM v0.3] MLP hidden  : {config.swiglu_hidden} (SwiGLU, 2/3 × {config.d_ff})")
        print(f"[TitanLM v0.3] GradCkpt    : {'ENABLED' if self._use_grad_ckpt else 'disabled'}")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids : (B, T) — token IDs
            labels    : (B, T) — target IDs; -100 positions are ignored in loss

        Returns:
            logits : (B, T, vocab_size)
            loss   : scalar cross-entropy (only when labels are provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Input length {T} exceeds model max_seq_len {self.config.max_seq_len}")

        x = self.embed_dropout(self.token_embedding(input_ids))

        for block in self.blocks:
            if self._use_grad_ckpt and self.training:
                x = gradient_checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x      = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def num_parameters(self, include_embeddings: bool = True) -> int:
        if include_embeddings:
            return sum(p.numel() for p in self.parameters())
        return sum(
            p.numel() for name, p in self.named_parameters()
            if "token_embedding" not in name
        )


# ─── Builder ──────────────────────────────────────────────────────────────────

def build_model(config: dict) -> TitanLM:
    """Build a TitanLM from a raw YAML config dict."""
    cfg = TitanConfig.from_dict(config)
    return TitanLM(cfg)
