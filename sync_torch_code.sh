#!/bin/bash

# Define source and destination paths
SOURCE_DIR="$HOME/pytorch/torch/_inductor"
DEST_DIR="./source-code-data/pytorch"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Use rsync to sync files
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose output
# --delete: remove files in destination that don't exist in source
# --exclude: exclude certain files/directories if needed
rsync -av "$SOURCE_DIR" "$DEST_DIR"

echo "Synchronization completed!"
