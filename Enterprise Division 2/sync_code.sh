#!/bin/bash

# Define the source and target directories
SOURCE_DIR="../Enterprise Division/"  # Adjust the path if necessary
TARGET_DIR="."  # Current directory (Enterprise Division 2 folder)

# Use rsync to copy everything from SOURCE_DIR to TARGET_DIR, excluding the .env file
rsync -av --exclude '.env' "$SOURCE_DIR" "$TARGET_DIR"

echo "Code synced from Enterprise Division #1 to Enterprise Division #2, excluding .env file."
