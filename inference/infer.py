#!/usr/bin/env python3
"""
TitanAI Inference Server
========================
Loads the trained 1B checkpoint and serves it as a REST API.
Your website's titanBuilder.ts route proxies requests to this server.

Usage:
  python inference/infer.py --serve                     # start API on port 8080
  python inference/infer.py --serve --port 9000         # custom port
  python inference/infer.py --prompt "Hello Titan"      # single inference (no server)

Endpoints:
  GET  /health       — check model status
  POST /generate     — generate a response
       body: { messages: [{role, content}], max_tokens: 512, temperature: 0.7 }
"""

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "titan_1b_pretrain" / "latest"
DEFAULT_CONFIG     = ROOT / "titan_1b.yaml"
DEFAULT_PORT       = 8080
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model loading ─────────────────────────────────────────────────────────────

_model     = None
_tokenizer = None
_config    = None
_lock      = threading.Lock()


def load_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback minimal YAML parser for key=value
        cfg = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    k, _, v = line.partition(":")
                    cfg[k.strip()] = v.strip().strip('"')
        return cfg


def load_model(checkpoint_path: str, config_path: str):
    global _model, _tokenizer, _config
    print(f"[Titan] Loading config from {config_path}...")
    _config = load_yaml(config_path)

    print(f"[Titan] Loading tokenizer...")
    from tokenizers import Tokenizer
    tokenizer_cfg = _config.get("tokenizer", {})
    tok_path_str  = tokenizer_cfg.get("path", "tokenizer/artifacts_v32k/tokenizer.json") \
                    if isinstance(tokenizer_cfg, dict) \
                    else "tokenizer/artifacts_v32k/tokenizer.json"
    tok_path = ROOT / tok_path_str
    _tokenizer = Tokenizer.from_file(str(tok_path))

    print(f"[Titan] Loading model from {checkpoint_path}...")
    # Import your model class — adjust path if different
    try:
        from model.titan_model import TitanModel
        model_cfg = _config.get("model", {})
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        _model = TitanModel(
            vocab_size  = int(model_cfg.get("vocab_size",  32000)),
            d_model     = int(model_cfg.get("d_model",     2048)),
            n_heads     = int(model_cfg.get("n_heads",     16)),
            n_kv_heads  = int(model_cfg.get("n_kv_heads",  4)),
            n_layers    = int(model_cfg.get("n_layers",    24)),
            d_ff        = int(model_cfg.get("d_ff",        8192)),
            max_seq_len = int(model_cfg.get("max_seq_len", 2048)),
            rope_base   = float(model_cfg.get("rope_base", 500000.0)),
            dropout     = 0.0,   # inference — no dropout
        )
    except ImportError as e:
        print(f"[Titan] ⚠  Could not import TitanModel: {e}")
        print("[Titan] Falling back to stub model for API testing.")
        _model = StubModel()
        return

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        candidates = sorted(ckpt_path.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"[Titan] ⚠  No .pt files found in {ckpt_path}. Using untrained weights.")
        else:
            ckpt_file = candidates[0]
            print(f"[Titan] Loading weights: {ckpt_file.name}")
            state = torch.load(ckpt_file, map_location=DEVICE, weights_only=True)
            model_state = state.get("model", state)
            _model.load_state_dict(model_state, strict=False)

    _model = _model.to(DEVICE).eval()
    param_count = sum(p.numel() for p in _model.parameters()) / 1e9
    print(f"[Titan] ✅ Model ready — {param_count:.2f}B params on {DEVICE.upper()}")


# ── Stub model for API testing without a checkpoint ───────────────────────────

class StubModel:
    """Returns placeholder responses so the website works even before training."""
    def to(self, *a, **k): return self
    def eval(self): return self
    def parameters(self): return iter([torch.zeros(1)])
    def __call__(self, *a, **k):
        return None


# ── Generation ────────────────────────────────────────────────────────────────

def format_prompt(messages: list) -> str:
    """Convert [{role, content}] to a flat prompt string."""
    parts = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


@torch.no_grad()
def generate(messages: list, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.95) -> str:
    if isinstance(_model, StubModel):
        return "Titan model is still training. Check back after training completes on Vast.ai!"

    prompt = format_prompt(messages)
    ids    = _tokenizer.encode(prompt).ids
    tokens = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    eos_id = _tokenizer.token_to_id("<|eos|>") or 2
    generated = []

    for _ in range(max_tokens):
        logits = _model(tokens)[:, -1, :]   # (1, vocab)

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature

        # Top-p (nucleus) sampling
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative - sorted_probs > top_p] = 0
        sorted_probs /= sorted_probs.sum()
        next_id = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)].item()

        if next_id == eos_id:
            break

        generated.append(next_id)
        tokens = torch.cat([tokens, torch.tensor([[next_id]], device=DEVICE)], dim=1)

        # Truncate context if too long
        if tokens.shape[1] > 2048:
            tokens = tokens[:, -2048:]

    return _tokenizer.decode(generated)


# ── HTTP handler ──────────────────────────────────────────────────────────────

class TitanHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        msg = (fmt % args) if args else fmt
        print(f"[Titan API] {msg}")

    def send_json(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            is_stub = isinstance(_model, StubModel)
            self.send_json(200, {
                "status":  "ok",
                "model":   "titan-1b",
                "device":  DEVICE,
                "trained": not is_stub,
                "version": "0.4.0",
            })
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/generate":
            self.send_json(404, {"error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        try:
            data        = json.loads(body)
            messages    = data.get("messages", [])
            max_tokens  = min(int(data.get("max_tokens",  512)), 1024)
            temperature = float(data.get("temperature", 0.7))
            top_p       = float(data.get("top_p",       0.95))
        except (json.JSONDecodeError, ValueError) as e:
            self.send_json(400, {"error": "Invalid request", "detail": str(e)})
            return

        if not messages:
            self.send_json(400, {"error": "messages array is required"})
            return

        with _lock:
            t0   = time.time()
            text = generate(messages, max_tokens, temperature, top_p)
            ms   = round((time.time() - t0) * 1000)

        self.send_json(200, {
            "text":          text,
            "model":         "titan-1b",
            "latency_ms":    ms,
            "finish_reason": "stop",
        })


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TitanAI Inference Server")
    parser.add_argument("--serve",      action="store_true",    help="Start HTTP server")
    parser.add_argument("--port",       type=int,  default=DEFAULT_PORT)
    parser.add_argument("--checkpoint", type=str,  default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config",     type=str,  default=str(DEFAULT_CONFIG))
    parser.add_argument("--prompt",     type=str,  default=None, help="Single prompt (no server)")
    args = parser.parse_args()

    load_model(args.checkpoint, args.config)

    if args.prompt:
        result = generate([{"role": "user", "content": args.prompt}])
        print("\nTitan:", result)
        return

    if args.serve:
        server = HTTPServer(("0.0.0.0", args.port), TitanHandler)
        print(f"[Titan] 🚀 Serving on http://0.0.0.0:{args.port}")
        print(f"[Titan]    Health: http://0.0.0.0:{args.port}/health")
        print(f"[Titan]    Press Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[Titan] Stopped.")

if __name__ == "__main__":
    main()


# ── TitanInference class (required by api/core/model_manager.py) ──────────────

class TitanInference:
    """
    Class wrapper around the module-level load_model / generate functions.
    model_manager.py imports and instantiates this class.
    """

    def __init__(self):
        self._loaded = False
        self._checkpoint: str = ""
        self._config_path: str = ""
        self._device: str = DEVICE
        self._param_count: float = 0.0

    def load(self, checkpoint_path: str, config_path: str) -> None:
        load_model(checkpoint_path, config_path)
        self._loaded = not isinstance(_model, StubModel)
        self._checkpoint = checkpoint_path
        self._config_path = config_path
        self._device = DEVICE
        if self._loaded:
            self._param_count = sum(p.numel() for p in _model.parameters()) / 1e9

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def checkpoint(self) -> str:
        return self._checkpoint

    @property
    def param_count(self) -> float:
        return self._param_count

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        **kwargs,
    ) -> str:
        return generate(messages, max_tokens=max_new_tokens,
                        temperature=temperature, top_p=top_p)

    def stream_generate(
        self,
        messages: list,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        **kwargs,
    ):
        """Token-by-token streaming generator."""
        global _model, _tokenizer
        if isinstance(_model, StubModel):
            yield "Titan model is still training. Check back after training completes!"
            return

        prompt = format_prompt(messages)
        ids    = _tokenizer.encode(prompt).ids
        tokens = torch.tensor([ids], dtype=torch.long, device=self._device)
        eos_id = _tokenizer.token_to_id("<|eos|>") or 2

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = _model(tokens)[:, -1, :]
                if temperature > 0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                sorted_probs[cumulative - sorted_probs > top_p] = 0
                sorted_probs /= sorted_probs.sum()
                next_id = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)].item()
                if next_id == eos_id:
                    break
                token_str = _tokenizer.decode([next_id])
                yield token_str
                tokens = torch.cat(
                    [tokens, torch.tensor([[next_id]], device=self._device)], dim=1)
                if tokens.shape[1] > 2048:
                    tokens = tokens[:, -2048:]
