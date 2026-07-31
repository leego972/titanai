#!/usr/bin/env bash
# TitanAI professional pack ingestion entrypoint.
# Usage: ./scripts/process_uploaded_training_pack.sh /path/to/uploaded-pack.zip
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PACK="${1:-}"
MANIFEST="${MANIFEST:-${REPO}/configs/training_packs/batch_01_core_engineering_security.json}"
OUTPUT="${OUTPUT:-${REPO}/data/professional/batch_01}"

if [[ -z "${PACK}" ]]; then
  echo "Usage: $0 /path/to/training-pack[.zip|.tar.gz]" >&2
  exit 64
fi
if [[ ! -e "${PACK}" ]]; then
  echo "Training pack not found: ${PACK}" >&2
  exit 66
fi

mkdir -p "${OUTPUT}"
python3 "${REPO}/scripts/compile_training_pack.py" \
  "${PACK}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT}"

if [[ ! -f "${OUTPUT}/READY" ]] || ! grep -qx "ready" "${OUTPUT}/READY"; then
  echo "Pack compilation did not produce a trainable dataset." >&2
  exit 65
fi

cat <<EOF
TitanAI Batch 1 pack compiled successfully.
Output: ${OUTPUT}
Report: ${OUTPUT}/compile_report.json
CPT: ${OUTPUT}/cpt.jsonl
SFT: ${OUTPUT}/sft.jsonl
DPO: ${OUTPUT}/dpo.jsonl
Evaluation holdout: ${OUTPUT}/evaluation.jsonl
Rejected records: ${OUTPUT}/rejected.jsonl
EOF
