#!/usr/bin/env bash
LIBRISPEECH_DIR="YOUR_LIBRISPEECH_DIR"
OUT_DIR="YOUR_OUT_DIR"
ONNX_PATH="YOUR_ONNX_PATH" # speech_tokenizer_v1.onnx under CosyVoice-300M model dir

mkdir -p "$OUT_DIR"
find "$LIBRISPEECH_DIR" -type f \( -iname "*.flac" -o -iname "*.wav" \) | sort | while read -r f; do
  rel="${f#"$LIBRISPEECH_DIR"/}"
  id="${rel%.*}"; id="${id//\//-}"
  echo "$id $f"
done > "$OUT_DIR/wav.scp"

python3 "YOUR_COSYVOICE_DIR/tools/extract_speech_token_mp.py" \
  --dir "$OUT_DIR" \
  --onnx_path "$ONNX_PATH"