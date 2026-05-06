#!/usr/bin/env bash
set -euo pipefail

# setup_score_sde.sh
# Downloads the public Score SDE PyTorch implementation and places the required
# files into ./score_sde/ so that the CIFAR-10 pruning/evaluation scripts can run.

REPO_URL="https://github.com/yang-song/score_sde_pytorch.git"
THIRD_PARTY_DIR="third_party/score_sde_pytorch"
TARGET_DIR="score_sde"

mkdir -p third_party

if [ ! -d "$THIRD_PARTY_DIR/.git" ]; then
  git clone "$REPO_URL" "$THIRD_PARTY_DIR"
else
  echo "[Info] $THIRD_PARTY_DIR already exists. Pulling latest changes..."
  git -C "$THIRD_PARTY_DIR" pull
fi

mkdir -p "$TARGET_DIR"

# Copy required Score-SDE source files.
cp -r "$THIRD_PARTY_DIR/models" "$TARGET_DIR/"
cp "$THIRD_PARTY_DIR/sde_lib.py" "$TARGET_DIR/"
cp "$THIRD_PARTY_DIR/losses.py" "$TARGET_DIR/"

# These files are useful for compatibility with the original implementation.
for f in sampling.py likelihood.py utils.py datasets.py evaluation.py; do
  if [ -f "$THIRD_PARTY_DIR/$f" ]; then
    cp "$THIRD_PARTY_DIR/$f" "$TARGET_DIR/"
  fi
done

# Ensure Python package markers exist.
touch "$TARGET_DIR/__init__.py"
touch "$TARGET_DIR/models/__init__.py"

echo "[Done] Score SDE files copied into $TARGET_DIR/"
echo "You can now run the CIFAR-10 pruning/evaluation scripts."
