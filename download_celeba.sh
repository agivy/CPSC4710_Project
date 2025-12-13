#!/bin/bash
# Set your base directory
BASE_DIR="/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/celeba"
mkdir -p $BASE_DIR
cd $BASE_DIR

echo "Downloading CelebA dataset..."

# Download Google Drive files using gdown (install: pip install gdown)
# CelebA images (img_align_celeba.zip) - ~1.3GB
echo "Downloading aligned face images..."
gdown --id 0B7EVK8r0v71pZjFTYXZWM3FlRnM -O img_align_celeba.zip

# Download attributes file
echo "Downloading attributes..."
gdown --id 0B7EVK8r0v71pblRyaVFSWGxPY0U -O list_attr_celeba.txt

# Download identity annotations
echo "Downloading identity annotations..."
gdown --id 1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS -O identity_CelebA.txt

# Download landmarks
echo "Downloading landmarks..."
gdown --id 0B7EVK8r0v71pd0FJY3Blby1HUTQ -O list_landmarks_align_celeba.txt

# Download bbox
echo "Downloading bounding boxes..."
gdown --id 0B7EVK8r0v71pbThiMVRxWXZ4dU0 -O list_bbox_celeba.txt

# Extract images
echo "Extracting images..."
unzip -q img_align_celeba.zip

# Clean up
# rm img_align_celeba.zip

echo "CelebA download complete!"
echo "Images are in: $BASE_DIR/img_align_celeba/"