#!/bin/bash

# FairFace Dataset Download Script
# This script downloads the FairFace dataset for fairness analysis

set -e  # Exit on error

# Set your base directory
BASE_DIR="/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/fairface"

echo "=========================================="
echo "FairFace Dataset Download Script"
echo "=========================================="
echo ""
echo "Base directory: $BASE_DIR"
echo ""

# Create directory structure
mkdir -p $BASE_DIR
cd $BASE_DIR

echo "Step 1: Downloading training images..."
echo "----------------------------------------"
# Download training images (approximately 4.3 GB)
wget -O train.zip "https://drive.google.com/uc?export=download&id=1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86&confirm=t"

echo ""
echo "Step 2: Downloading validation images..."
echo "----------------------------------------"
# Download validation images (approximately 1.1 GB)
wget -O val.zip "https://drive.google.com/uc?export=download&id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH&confirm=t"

echo ""
echo "Step 3: Downloading training labels..."
echo "----------------------------------------"
# Download training labels CSV
wget -O fairface_label_train.csv "https://raw.githubusercontent.com/dchen236/FairFace/master/fairface_label_train.csv"

echo ""
echo "Step 4: Downloading validation labels..."
echo "----------------------------------------"
# Download validation labels CSV
wget -O fairface_label_val.csv "https://raw.githubusercontent.com/dchen236/FairFace/master/fairface_label_val.csv"

echo ""
echo "Step 5: Extracting training images..."
echo "----------------------------------------"
unzip -q train.zip
echo "Training images extracted to: $BASE_DIR/train/"

echo ""
echo "Step 6: Extracting validation images..."
echo "----------------------------------------"
unzip -q val.zip
echo "Validation images extracted to: $BASE_DIR/val/"

echo ""
echo "Step 7: Cleaning up zip files..."
echo "----------------------------------------"
rm train.zip val.zip
echo "Zip files removed."

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Dataset structure:"
echo "$BASE_DIR/"
echo "  ├── train/           ($(find train -name '*.jpg' | wc -l) images)"
echo "  ├── val/             ($(find val -name '*.jpg' | wc -l) images)"
echo "  ├── fairface_label_train.csv"
echo "  └── fairface_label_val.csv"
echo ""
echo "Dataset statistics:"
wc -l fairface_label_train.csv
wc -l fairface_label_val.