# TitanAI — Cost-Optimised Training Plan
  Updated: May 2026 | Target: Production-quality 1B → optionally 3B

  ## The Goal Restated

  Two deployment targets:
  - **Archibald** (Cybersecurity): complex threat modelling, CVE analysis, secure architecture,
    exploit research, defensive tool building. Needs deep technical reasoning, not just surface recall.
  - **Virelle** (Cinema): production intelligence, script analysis, scheduling, creative direction.

  These are domain-specific expert assistants — NOT generic chatbots. This changes the training
  strategy significantly: a well-trained 1B beats a poorly-trained 3B for specialised tasks.

  ---

  ## GPU Cost Reference (Vast.ai Spot, May 2026)

  | GPU           | VRAM  | $/hr  | 1B tok/s | Notes                          |
  |---------------|-------|-------|----------|--------------------------------|
  | RTX 4090      | 24 GB | $0.40 | ~80K     | Best $/token for 1B model      |
  | 2× RTX 4090   | 48 GB | $0.80 | ~150K    | DDP: 10% cheaper per token     |
  | A100 40GB SXM | 40 GB | $1.40 | ~200K    | More VRAM, not faster per $    |
  | A100 80GB SXM | 80 GB | $2.00 | ~280K    | Needed for 3B without grad ckpt|
  | H100 SXM      | 80 GB | $3.50 | ~550K    | Best for 3B if budget allows   |

  Tokens/sec assumes: Flash Attention 2 + bf16 + torch.compile + 8-bit Adam ON.
  Without Flash Attention 2, divide by ~1.4.

  ---

  ## Phase Plan + Costs

  ### Phase 1: 1B Pretraining (9.8B tokens via streaming)
  Script: `pretrain_titan_v3.py` — streams HuggingFace datasets, no local corpus needed.
  Config: `configs/titan_1b.yaml` (max_steps=150,000)

  Effective batch = 2 × 32 grad_accum × 2048 seq_len = 131,072 tokens/step
  150,000 steps × 131,072 = **9.83B tokens**

  | GPU           | Steps/hr | hrs    | Cost     | Verdict          |
  |---------------|----------|--------|----------|------------------|
  | RTX 4090      | ~3,300   | 45.5   | **$18**  | ✅ Best value     |
  | 2× RTX 4090   | ~6,200   | 24.2   | **$19**  | ✅ Same cost, 2×  |
  | A100 40GB     | ~5,500   | 27.3   | **$38**  | ❌ 2× cost, same  |

  **Run on RTX 4090 or 2× RTX 4090 (same price, faster)**

  Minimum viable (6B tokens, 91,500 steps): **~$11** — gets you ~80% of peak quality.
  Recommendation: run to 91,500, evaluate, extend if quality gates pass.

  ---

  ### Phase 2: SFT v1 — Instruction Format
  Script: `scripts/sft_train.py` or `scripts/run_sft_v2.py`
  Config: `configs/titan_sft_v01.yaml` / `titan_sft_v02.yaml`

  ~50,000 steps on RTX 4090 = ~6 hrs = **$2.40**

  ---

  ### Phase 3: Domain Fine-tuning (Cyber + Cinema)
  Config: `configs/titan_cyber_deepdive.yaml` (50,000 steps, 4K context)

  RTX 4090 can handle 3B context with grad checkpointing but NOT 4K on 1B without it.
  With grad checkpointing on 1B @ 4K context on RTX 4090:
  ~50,000 steps × ~3.5 sec = ~49 hrs = **$20**

  Or on A100 40GB (more VRAM, no grad ckpt needed): ~18 hrs = **$25**

  Use RTX 4090 + gradient_checkpointing: true in config.

  ---

  ### Phase 4 (Optional): Scale to 3B
  Only do this if 1B quality gates pass AND you need more capacity.

  Steps:
  1. Run `python scripts/upscale_to_3b.py --src_checkpoint checkpoints/titan_1b_pretrain/final.pt`
  2. Continue pretraining 3B for ~3B more tokens (46,000 steps @ 65K tok/step)

  | GPU           | hrs  | Cost     | Notes                        |
  |---------------|------|----------|------------------------------|
  | A100 80GB     | ~30  | **$60**  | Needs 80GB for 3B w/o grad ckpt |
  | H100 SXM      | ~15  | **$53**  | Fastest, similar cost         |
  | 2× A100 40GB  | ~22  | **$62**  | DDP works, more setup        |

  **Do NOT attempt 3B on RTX 4090 without DDP** — 2.9B model + 8-bit Adam exceeds 24GB VRAM.

  ---

  ## Total Cost Summary

  ### Minimum (6B token base, no 3B):
  | Phase           | Cost  |
  |-----------------|-------|
  | 1B pretraining  | $11   |
  | SFT v1          | $2    |
  | Cyber deepdive  | $15   |
  | **Total**       | **$28** |

  ### Recommended (10B token base, no 3B):
  | Phase           | Cost  |
  |-----------------|-------|
  | 1B pretraining  | $18   |
  | SFT v1          | $2    |
  | SFT v2          | $3    |
  | Cyber deepdive  | $15   |
  | **Total**       | **$38** |

  ### Full pipeline (10B base + 3B scale-up):
  | Phase           | Cost  |
  |-----------------|-------|
  | 1B pipeline     | $38   |
  | 3B continuation | $55   |
  | **Total**       | **~$93** |

  ---

  ## Cost Reduction Techniques (Already Implemented)

  1. **Streaming pretraining** — no local corpus, no shard generation, no Dropbox bandwidth
     costs during training. Data comes from HuggingFace directly. Saves ~$3-5 in wasted
     GPU billing time per run.

  2. **8-bit Adam** (--use-8bit-adam) — halves optimizer memory. On 1B this saves ~12GB VRAM,
     allowing larger batch size on RTX 4090 → fewer steps needed for same token count.

  3. **torch.compile** (--compile) — ~18-22% throughput improvement with zero quality loss.
     On a 45-hour 1B run, this saves ~8-10 hours = ~$4.

  4. **FlashAttention-2** — ~35-40% throughput vs SDPA fallback. Wheel is now cached to
     Dropbox after first build, so every subsequent run skips the 10-minute compile.

  5. **Spot instances** — already using these via Vast.ai. Checkpoints sync to Dropbox on
     shutdown so interruptions don't lose progress.

  6. **Gradient checkpointing** — enables training on cheaper/smaller GPUs by trading
     ~15% compute for significant VRAM savings. Essential for 4K context fine-tuning on 4090.

  ---

  ## Quickstart (Today)

  ```bash
  # Step 1: Generate 1B init checkpoint from your 109M checkpoint
  python scripts/upscale_to_1b.py \
      --src_checkpoint /path/to/titan_109m/best.pt \
      --dst_checkpoint checkpoints/titan_1b_pretrain/init.pt

  # Step 2: Upload init checkpoint to Dropbox
  rclone copy checkpoints/titan_1b_pretrain/init.pt dropbox:TitanAI/checkpoints/titan_1b_pretrain/

  # Step 3: Start a Vast.ai RTX 4090 spot instance with these env vars:
  #   TITAN_REQUIRE_AUTH=true
  #   TITAN_API_KEY=<32+ char string>
  #   DROPBOX_RCLONE_TOKEN=<base64 of rclone.conf>
  #   TITAN_CONFIG=configs/titan_1b.yaml
  #   TITAN_CHECKPOINT_DIR=checkpoints/titan_1b_pretrain

  # Step 4: Set the on-start command to:
  #   bash /workspace/titanai/scripts/vast_startup.sh
  ```

  The instance will auto-configure, auto-resume from the latest checkpoint, train until done,
  push checkpoints to Dropbox, and shut itself down.
  