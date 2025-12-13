#!/bin/bash

# Set your base directory
BASE_DIR="/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/coco2017"  # UPDATE THIS
mkdir -p $BASE_DIR
cd $BASE_DIR

# Download training images (18GB)
echo "Downloading training images..."
wget http://images.cocodataset.org/zips/train2017.zip

# Download validation images (1GB)
echo "Downloading validation images..."
wget http://images.cocodataset.org/zips/val2017.zip

# Download annotations (241MB)
echo "Downloading annotations..."
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract all
echo "Extracting files..."
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# Clean up zip files (optional)
# rm *.zip

echo "Download complete!"