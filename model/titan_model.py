"""
Titan Base Language Model — v0.2
=================================
A decoder-only Transformer (GPT-style) implemented from scratch in PyTorch.

Architecture upgrades from Foundation v1:
    - Rotary Positional Embeddings (RoPE) replacing learned absolute embeddings
      Required for 2048+ context length and length extrapolation.
    - FlashAttention-2 integration with clean CPU/standard-attention fallback
      Active when flash_attn is installed and CUDA is available.
      Falls back gracefully to standard scaled dot-product attention otherwise.
    - Compatibility check at init: reports which attention path is active.

Locked from Foundation v1 (do not change without approval):
    - Pre-LayerNorm for training stability
    - GELU activation in MLP
    - Causal (autoregressive) masking
    - Optional embedding weight tying
    - Weight initialization scheme (scaled residual projections)

This is Titan's own architecture. It is not a wrapper over any external model.
All weights are initialized from scratch and owned by this project.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# ─── FlashAttention-2 availability check ──────────────────────────────────────

def _check_flash_attn() -> bool:
    """
    Returns True if flash_attn is installed AND CUDA is available.
    Prints a clear status message at import time.
    """
    try:
        import flash_attn  # noqa: F401
        if torch.cuda.is_available():
            print("[TitanLM] FlashAttention-2: ACTIVE (flash_attn installed, CUDA available)")
            return True
        else:
            print("[TitanLM] FlashAttention-2: INACTIVE (flash_attn installed but no CUDA — using standard attention)")
            return False
    except ImportError:
        print("[TitanLM] FlashAttention-2: INACTIVE (flash_attn not installed — using standard attention)")
        return False

FLASH_ATTN_AVAILABLE = _check_flash_attn()


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class TitanConfig:
    """All model hyperparameters. Load from titan_config.yaml via from_dict()."""
    vocab_size: int = 8000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 2048      # Upgraded: 2048 context supported via RoPE
    dropout: float = 0.1
    tie_embeddings: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "TitanConfig":
        m = cfg["model"]
        return cls(
            vocab_size=m["vocab_size"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            n_layers=m["n_layers"],
            d_ff=m["d_ff"],
            max_seq_len=m["max_seq_len"],
            dropout=m["dropout"],
            tie_embeddings=m["tie_embeddings"],
        )

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "tie_embeddings": self.tie_embeddings,
        }


# ─── Rotary Positional Embeddings (RoPE) ──────────────────────────────────────

def precompute_rope_freqs(d_head: int, max_seq_len: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE cosine and sine frequency tables.

    Args:
        d_head: dimension per attention head (must be even)
        max_seq_len: maximum sequence length
        base: frequency base (default 10000, as in original RoPE paper)

    Returns:
        cos_table: (max_seq_len, d_head)
        sin_table: (max_seq_len, d_head)
    """
    assert d_head % 2 == 0, f"d_head must be even for RoPE, got {d_head}"
    # Frequencies: theta_i = 1 / (base ^ (2i / d_head))
    theta = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))  # (d_head/2,)
    positions = torch.arange(max_seq_len).float()  # (max_seq_len,)
    freqs = torch.outer(positions, theta)  # (max_seq_len, d_head/2)
    # Interleave: repeat each freq for cos and sin pairing
    freqs_full = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, d_head)
    return freqs_full.cos(), freqs_full.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension to implement RoPE rotation."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Positional Embeddings to query and key tensors.

    Args:
        q: (B, n_heads, T, d_head)
        k: (B, n_heads, T, d_head)
        cos: (T, d_head) — precomputed cosines
        sin: (T, d_head) — precomputed sines

    Returns:
        q_rot, k_rot: same shape as input, with RoPE applied
    """
    # Reshape cos/sin to broadcast: (1, 1, T, d_head)
    cos = cos[:q.shape[2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:q.shape[2], :].unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ─── Attention ────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE positional embeddings.

    Attention path selection (in priority order):
        1. FlashAttention-2 (flash_attn installed + CUDA available)
        2. PyTorch scaled_dot_product_attention with causal mask (PyTorch >= 2.0)
        3. Manual scaled dot-product attention (fallback for all environments)
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, \
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"

        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len

        # Combined Q, K, V projection
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute RoPE frequency tables and register as buffers (not parameters)
        cos_table, sin_table = precompute_rope_freqs(self.d_head, config.max_seq_len)
        self.register_buffer("rope_cos", cos_table, persistent=False)
        self.register_buffer("rope_sin", sin_table, persistent=False)

        # Causal mask for manual attention fallback
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len),
        )

        # Determine which attention path to use
        self._use_flash = FLASH_ATTN_AVAILABLE
        self._use_sdpa = (not self._use_flash) and hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, d_model

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = apply_rope(q, k, self.rope_cos, self.rope_sin)

        # ── Attention path ──────────────────────────────────────────────────
        if self._use_flash:
            # FlashAttention-2: expects (B, T, n_heads, d_head) layout
            from flash_attn import flash_attn_func
            q_fa = q.transpose(1, 2)  # (B, T, n_heads, d_head)
            k_fa = k.transpose(1, 2)
            v_fa = v.transpose(1, 2)
            out = flash_attn_func(q_fa, k_fa, v_fa, causal=True)  # (B, T, n_heads, d_head)
            out = out.reshape(B, T, C)

        elif self._use_sdpa:
            # PyTorch 2.0+ native scaled_dot_product_attention (fused, efficient)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )  # (B, n_heads, T, d_head)
            out = out.transpose(1, 2).contiguous().view(B, T, C)

        else:
            # Manual fallback: standard scaled dot-product attention
            scale = math.sqrt(self.d_head)
            attn = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)
            attn = attn.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = attn @ v  # (B, n_heads, T, d_head)
            out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.resid_dropout(self.out_proj(out))
        return out


# ─── MLP ──────────────────────────────────────────────────────────────────────

class TitanMLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


# ─── Transformer Block ────────────────────────────────────────────────────────

class TitanBlock(nn.Module):
    """
    Single Transformer decoder block.
    Uses Pre-LayerNorm for training stability (locked from Foundation v1).
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = TitanMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ─── Full Model ───────────────────────────────────────────────────────────────

class TitanLM(nn.Module):
    """
    Titan Language Model — decoder-only Transformer.
    Foundation v1 architecture + RoPE + FlashAttention-2 path.
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.config = config

        # Token embedding only — positional info is handled by RoPE in attention
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TitanBlock(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Optionally tie input embedding weights to output projection
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) for stability
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

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
    ):
        """
        Args:
            input_ids: (B, T) token IDs
            labels:    (B, T) token IDs for loss computation, -100 = ignore

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss (only if labels provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        # Token embeddings only (RoPE handles positional information)
        x = self.embed_dropout(self.token_embedding(input_ids))

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        Args:
            input_ids:      (1, T) prompt token IDs
            max_new_tokens: number of tokens to generate
            temperature:    sampling temperature (1.0 = no scaling)
            top_k:          keep only top-k logits (0 = disabled)
            top_p:          nucleus sampling threshold (1.0 = disabled)
            eos_id:         stop generation when this token is produced
        Returns:
            (1, T + max_new_tokens) token IDs
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            ctx = generated[:, -self.config.max_seq_len:]
            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :]

            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_idx_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_idx_to_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        return generated


def build_model(config_dict: dict) -> TitanLM:
    """Convenience function: build TitanLM from the full config dict."""
    config = TitanConfig.from_dict(config_dict)
    model = TitanLM(config)
    n_params = model.count_parameters()
    print(f"[Model] TitanLM initialized: {n_params:,} parameters")
    print(f"[Model] Config: {config.n_layers} layers, {config.n_heads} heads, "
          f"d_model={config.d_model}, d_ff={config.d_ff}, "
          f"vocab={config.vocab_size}, ctx={config.max_seq_len}")
    return model
