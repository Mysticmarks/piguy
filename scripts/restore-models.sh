#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="$ROOT_DIR/static/models/transformers"
FORCE=0

if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

fetch_if_missing() {
  local url="$1"
  local out="$2"

  if [[ -f "$out" && "$FORCE" -eq 0 ]]; then
    echo "[skip] $out"
    return
  fi

  mkdir -p "$(dirname "$out")"
  echo "[get ] $url"
  curl -fsSL "$url" -o "$out"
}

SENTIMENT_CDN_BASE="https://huggingface.co/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main"
EMBED_CDN_BASE="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main"

SENTIMENT_BASE="$MODEL_ROOT/Xenova/distilbert-base-uncased-finetuned-sst-2-english"
EMBED_BASE="$MODEL_ROOT/Xenova/all-MiniLM-L6-v2"

SENTIMENT_FILES=(
  "config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "special_tokens_map.json"
  "onnx/model_quantized.onnx"
)

EMBED_FILES=(
  "config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "special_tokens_map.json"
  "onnx/model_quantized.onnx"
)

for file in "${SENTIMENT_FILES[@]}"; do
  fetch_if_missing \
    "$SENTIMENT_CDN_BASE/$file" \
    "$SENTIMENT_BASE/$file"
done

for file in "${EMBED_FILES[@]}"; do
  fetch_if_missing \
    "$EMBED_CDN_BASE/$file" \
    "$EMBED_BASE/$file"
done


VENDOR_DIR="$ROOT_DIR/static/vendor"
fetch_if_missing \
  "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js" \
  "$VENDOR_DIR/transformers.min.js"

echo "Model restore complete in: $MODEL_ROOT"
