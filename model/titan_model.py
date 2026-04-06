"""
Titan Base Language Model
=========================
A decoder-only Transformer (GPT-style) implemented from scratch in PyTorch.

Architecture:
    - Token embedding + learned positional embedding
    - N stacked Transformer decoder blocks
    - Each block: Pre-LayerNorm, Causal Multi-Head Self-Attention, Pre-LayerNorm, MLP
    - Final LayerNorm + linear projection to vocabulary
    - Optional weight tying between input embedding and output projection

This is Titan's own architecture. It is not a wrapper over any external model.
All weights are initialized from scratch and owned by this project.

Design decisions:
    - Pre-LayerNorm (before attention/MLP) for training stability
    - Causal attention mask (autoregressive, decoder-only)
    - GELU activation in the MLP
    - Optional embedding weight tying to reduce parameter count
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class TitanConfig:
    """All model hyperparameters. Load from titan_config.yaml via from_dict()."""
    vocab_size: int = 8000
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
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


# ─── Attention ────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    Uses a causal mask so each position can only attend to itself and earlier positions.
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, \
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"

        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model

        # Combined Q, K, V projection
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask (lower triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq_len, d_model

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape to (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        # Apply causal mask
        attn = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf"),
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
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
    Uses Pre-LayerNorm for better training stability.
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = TitanMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ─── Full Model ───────────────────────────────────────────────────────────────

class TitanLM(nn.Module):
    """
    Titan Language Model — decoder-only Transformer.
    This is Titan's own base model, trained from scratch.
    """

    def __init__(self, config: TitanConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
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
            labels: (B, T) token IDs for loss computation, -100 = ignore

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss (only if labels provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        device = input_ids.device
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        # Embeddings
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.embed_dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Flatten for cross-entropy
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
            input_ids: (1, T) prompt token IDs
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (1.0 = no scaling)
            top_k: keep only top-k logits (0 = disabled)
            top_p: nucleus sampling threshold (1.0 = disabled)
            eos_id: stop generation when this token is produced
        Returns:
            (1, T + max_new_tokens) token IDs
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max context window
            ctx = generated[:, -self.config.max_seq_len:]
            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :]  # Last token logits: (1, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above threshold
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
