# Fairness-Aware Cross-Modal Interpretability for Vision-Language Models

This repository contains the implementation and evaluation code for training and assessing Vision-Language Models (VLMs) with a focus on fairness, interpretability, and trustworthiness.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset Setup](#dataset-setup)
- [Usage Guide](#usage-guide)
- [Reproducibility Guide](#reproducibility-guide)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements several approaches to training and evaluating VLMs:

1. **Baseline VLM Training**: Fine-tuning BLIP on FairFace dataset with demographic annotations
2. **ACMFO (Adaptive Cross-Modal Fairness Optimizer)**: Multi-objective training on MS-COCO with fairness constraints
3. **Standard BLIP Training**: Training on FairFace with demographic-aware captions
4. **Comprehensive Evaluation**: Bias analysis, fairness metrics, and performance evaluation

### Key Features
- Demographic parity constraint via KL divergence
- Cross-modal disentanglement via mutual information minimization
- CLIP-based demographic inference for datasets without labels
- Multiple evaluation metrics (BLEU, ROUGE-L, fairness metrics)
- Comprehensive bias analysis across gender, race, and age

---

## Project Structure

### Training Scripts
- **`acmfo_training.py`** - Main implementation of the Adaptive Cross-Modal Fairness Optimizer (ACMFO) framework for training BLIP on MS-COCO with fairness constraints, cross-modal disentanglement, and demographic parity regularization
- **`baseline_vlm_training.py`** - Baseline VLM fine-tuning script that trains GIT model on FairFace dataset with demographic annotations (age, gender, race)
- **`train_blip.py`** - Standard BLIP training pipeline on FairFace dataset, automatically downloads data from Hugging Face and trains with demographic-aware caption generation

### Dataset Download Scripts
- **`download_coco.sh`** - Bash script to download MS-COCO 2017 dataset (train images ~18GB, val images ~1GB, annotations ~241MB) and extract to specified directory
- **`download_fairface.sh`** - Bash script to download FairFace dataset from Google Drive including training images (~4.3GB), validation images (~1.1GB), and CSV label files
- **`download_fairface.py`** - Python alternative for downloading FairFace using Hugging Face datasets library (automatic caching)
- **`download_celeba.sh`** - Bash script to download CelebA dataset using gdown for Google Drive files (aligned faces ~1.3GB, attributes, landmarks, bounding boxes)

### Evaluation Scripts
- **`evaluate_vlm.py`** - Comprehensive trustworthiness evaluation framework that measures fairness (DPD), interpretability (attention entropy, concentration), reliability (ECE), and performance (BLEU, ROUGE) on pretrained VLMs using MS-COCO validation set
- **`evaluate_train_blip.py`** - Evaluation script specifically for fine-tuned BLIP models, computes BLEU/ROUGE scores, demographic-stratified performance, bias amplification metrics, and confusion matrices
- **`causal_tracing.py`** - Implements causal intervention analysis to measure faithfulness of attention mechanisms by corrupting visual patches with noise, selectively restoring them, and computing correlation between attention weights and actual causal impact

### Jupyter Notebooks
- **`vlm_train-2.ipynb`** - Main interactive training notebook with end-to-end pipeline: dataset loading, ACMFO training, loss monitoring, and preliminary results visualization
- **`baseline_vlm_training.ipynb`** - Interactive notebook for baseline VLM experiments on FairFace/CelebA-HQ with demographic fairness analysis and training curve visualization
- **`coco_bias_analysis-3.ipynb`** - Comprehensive bias analysis notebook for MS-COCO dataset: gender distribution statistics, object-gender associations, activity bias patterns, word frequency analysis, and visualization generation

---

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM for full training)
- 50GB+ free disk space for datasets
- Linux/Unix environment (for bash scripts)

### Python Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Core dependencies include:
- `torch>=2.0.0` - PyTorch deep learning framework
- `torchvision>=0.15.0` - Vision utilities
- `transformers>=4.35.0` - Hugging Face transformers (BLIP, CLIP, etc.)
- `datasets>=2.14.0` - Hugging Face datasets library
- `accelerate>=0.24.0` - Distributed training
- `pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualizations
- `scikit-learn>=1.3.0` - Machine learning utilities
- `scipy>=1.11.0` - Scientific computing
- `nltk>=3.8.0` - Natural language toolkit
- `rouge-score>=0.1.2` - ROUGE metric computation

### Additional Dependencies

Some scripts require additional packages:

```bash
# For COCO evaluation
pip install pycocotools

# For CelebA download (optional)
pip install gdown

# For causal tracing
pip install captum  # If not already installed
```

Download NLTK data:
```python
import nltk
nltk.download('punkt')
```

---

## Dataset Setup

The project uses three main datasets. **IMPORTANT**: Update all dataset paths in the scripts before running!

### 1. MS-COCO 2017 Dataset

**Used in**: `acmfo_training.py`, `evaluate_vlm.py`, bias analysis notebooks

**Default Path**: `/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/coco2017`

**Download Instructions**:

#### Option A: Using the provided script (recommended)

```bash
# 1. Edit download_coco.sh and update BASE_DIR
nano download_coco.sh
# Change: BASE_DIR="/your/path/to/coco2017"

# 2. Run the download script
chmod +x download_coco.sh
./download_coco.sh
```

#### Option B: Manual download

```bash
# Create directory
mkdir -p /your/path/to/coco2017
cd /your/path/to/coco2017

# Download images (~19GB total)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Download annotations (~241MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip

# Optional: Remove zips to save space
rm *.zip
```

**Expected Structure**:
```
coco2017/
├── train2017/          # 118,287 training images
├── val2017/            # 5,000 validation images
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    ├── captions_train2017.json
    └── captions_val2017.json
```

**Update Paths In**:
- `acmfo_training.py` → Line 58: `COCO_BASE_DIR`
- `evaluate_vlm.py` → Line 44: `COCO_BASE_DIR`
- `download_coco.sh` → Line 4: `BASE_DIR`

---

### 2. FairFace Dataset

**Used in**: `train_blip.py`, `baseline_vlm_training.py`

**Default Path**: `/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/fairface`

**Download Instructions**:

#### Option A: Using Hugging Face (recommended - automatic)

The `train_blip.py` script automatically downloads FairFace from Hugging Face:

```python
# This is handled automatically in the script
from datasets import load_dataset
dataset = load_dataset("HuggingFaceM4/FairFace", "0.25", cache_dir=FAIRFACE_BASE_DIR)
```

Just update the path in the script:
```bash
# Edit train_blip.py
nano train_blip.py
# Line 31: FAIRFACE_BASE_DIR = "/your/path/to/fairface"
# Line 92: FAIRFACE_BASE_DIR = "/your/path/to/fairface"
```

#### Option B: Manual download using bash script

```bash
# 1. Edit download_fairface.sh
nano download_fairface.sh
# Line 9: BASE_DIR="/your/path/to/fairface"

# 2. Run the script
chmod +x download_fairface.sh
./download_fairface.sh
```

**Dataset Details**:
- Training images: ~86,000 face images
- Validation images: ~10,000 face images
- Attributes: Age (7 bins), Gender (Male/Female), Race (7 categories)
- Image size: 224x224 (face-cropped)

**Expected Structure** (Hugging Face):
```
fairface/
├── downloads/          # Cached dataset files
└── HuggingFaceM4___fair_face/
    └── 0.25/
        ├── train/      # Training split
        └── validation/ # Validation split
```

**Expected Structure** (Manual):
```
fairface/
├── train/              # Training images
├── val/                # Validation images
├── fairface_label_train.csv
└── fairface_label_val.csv
```

**Update Paths In**:
- `train_blip.py` → Lines 31, 92: `FAIRFACE_BASE_DIR`
- `baseline_vlm_training.py` → Line 69: `base_dir`
- `download_fairface.sh` → Line 9: `BASE_DIR`

---

### 3. CelebA Dataset (Optional)

**Used in**: Some baseline experiments (optional)

**Default Path**: `/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/celeba`

**Download Instructions**:

```bash
# 1. Install gdown for Google Drive downloads
pip install gdown

# 2. Edit download_celeba.sh
nano download_celeba.sh
# Line 3: BASE_DIR="/your/path/to/celeba"

# 3. Run the script
chmod +x download_celeba.sh
./download_celeba.sh
```

**Dataset Details**:
- 202,599 celebrity face images
- 40 binary attributes per image
- Image size: 178x218 (aligned and cropped)

**Expected Structure**:
```
celeba/
├── img_align_celeba/       # Aligned face images
├── list_attr_celeba.txt    # Attributes
├── identity_CelebA.txt     # Identity labels
├── list_landmarks_align_celeba.txt
└── list_bbox_celeba.txt
```

---

## Usage Guide

### Training Workflows

#### 1. ACMFO Training (Main Method)

The ACMFO framework implements fairness-aware training with cross-modal disentanglement:

```bash
# Edit paths in the script
nano acmfo_training.py
# Line 58: COCO_BASE_DIR = "/your/path/to/coco2017"
# Line 59: OUTPUT_DIR = "acmfo_coco_checkpoints"

# Run training
python acmfo_training.py
```

**Key Configuration Parameters** (lines 54-87):
- `BATCH_SIZE = 8` - Adjust based on GPU memory
- `EPOCHS = 3` - Number of training epochs
- `TRAIN_SAMPLES = 20000` - Use subset for faster training
- `LAMBDA_FAIRNESS = 1.0` - Fairness loss weight
- `LAMBDA_CROSS_MODAL = 0.05` - Cross-modal disentanglement weight
- `NUM_CAPTION_CLUSTERS = 100` - k-means clusters for KL approximation

**Outputs**: Checkpoints saved to `acmfo_coco_checkpoints/` with training logs and fairness metrics

#### 2. Baseline VLM Training

Train GIT model on FairFace dataset:

```bash
# Edit paths
nano baseline_vlm_training.py
# Line 69: base_dir = "/your/path/to/project"

# Run training
python baseline_vlm_training.py
```

**Outputs**: Model checkpoints, training curves, and demographic distribution plots in `baseline_vlm/`

#### 3. BLIP Training on FairFace

Standard BLIP fine-tuning with automatic dataset download:

```bash
# Edit paths
nano train_blip.py
# Line 31: FAIRFACE_BASE_DIR = "/your/path/to/fairface"
# Line 92: FAIRFACE_BASE_DIR = "/your/path/to/fairface"

# Run training
python train_blip.py
```

**Special Features**:
- Automatically downloads FairFace from Hugging Face
- Generates demographic-aware captions using age, gender, and race attributes
- Uses larger batch size (32) since FairFace images are smaller

**Outputs**: Fine-tuned model in `checkpoints/blip-finetuned-fairface/`

### Evaluation Workflows

#### 1. Comprehensive VLM Evaluation

Evaluate any VLM on trustworthiness metrics:

```bash
# Edit paths
nano evaluate_vlm.py
# Line 44: COCO_BASE_DIR = "/your/path/to/coco2017"

# Run evaluation
python evaluate_vlm.py
```

**Metrics Computed**:
- Performance: BLEU-1 to BLEU-4, ROUGE-L
- Fairness: Demographic Parity Difference (DPD)
- Interpretability: Attention entropy, concentration score
- Reliability: Expected Calibration Error (ECE)

**Outputs**: Results in `vlm_trust_evaluation/` with CSV files and visualizations

#### 2. BLIP-Specific Evaluation

Evaluate fine-tuned BLIP models:

```bash
python evaluate_train_blip.py \
    --model_path checkpoints/blip-finetuned-fairface \
    --dataset_path /path/to/fairface \
    --output_dir evaluation_results
```

#### 3. Causal Tracing Analysis

Measure attention faithfulness via causal interventions:

```bash
python causal_tracing.py \
    --model_path checkpoints/acmfo_coco_checkpoints/final_model \
    --image_path sample_images/test_image.jpg \
    --output_dir causal_results
```

**Analysis Process**:
1. Generate baseline caption from clean image
2. Corrupt all patches with Gaussian noise
3. Restore each patch individually
4. Measure probability change per token
5. Compute Spearman correlation between attention weights and causal impact

### Interactive Notebooks

#### 1. Main Training Notebook

```bash
jupyter notebook vlm_train-2.ipynb
```

End-to-end pipeline for ACMFO training with interactive monitoring

#### 2. Baseline Experiments

```bash
jupyter notebook baseline_vlm_training.ipynb
```

Baseline VLM training on FairFace/CelebA-HQ with fairness analysis

#### 3. COCO Bias Analysis

```bash
jupyter notebook coco_bias_analysis-3.ipynb
```

Comprehensive dataset bias analysis with statistical visualizations

---

## Reproducibility Guide

### Step-by-Step Reproduction

#### Phase 1: Environment Setup (15 minutes)

```bash
# 1. Clone/setup repository
mkdir vlm_fairness_project
cd vlm_fairness_project

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install additional packages
pip install pycocotools gdown

# 5. Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

#### Phase 2: Dataset Preparation

```bash
# 1. Create dataset directories
mkdir -p datasets/{coco2017,fairface,celeba}

# 2. Download MS-COCO (required for ACMFO and evaluation)
# Edit download_coco.sh first!
nano download_coco.sh  # Update BASE_DIR
chmod +x download_coco.sh
./download_coco.sh
# Expected time: 30-60 minutes
# Expected size: ~20GB

# 3. FairFace will be downloaded automatically by train_blip.py
# Or manually:
nano download_fairface.sh  # Update BASE_DIR
chmod +x download_fairface.sh
./download_fairface.sh
# Expected time: 20-40 minutes
# Expected size: ~5.5GB

# 4. CelebA (optional)
nano download_celeba.sh  # Update BASE_DIR
chmod +x download_celeba.sh
./download_celeba.sh
# Expected time: 15-30 minutes
```

#### Phase 3: Update All Paths

Create a script to update paths automatically:

```bash
# update_paths.sh
#!/bin/bash

# Set your base directory
YOUR_BASE_DIR="/your/actual/path/here"
COCO_PATH="${YOUR_BASE_DIR}/datasets/coco2017"
FAIRFACE_PATH="${YOUR_BASE_DIR}/datasets/fairface"
CELEBA_PATH="${YOUR_BASE_DIR}/datasets/celeba"

# Update acmfo_training.py
sed -i "s|COCO_BASE_DIR = .*|COCO_BASE_DIR = \"${COCO_PATH}\"|" acmfo_training.py

# Update evaluate_vlm.py
sed -i "s|COCO_BASE_DIR = .*|COCO_BASE_DIR = \"${COCO_PATH}\"|" evaluate_vlm.py

# Update train_blip.py (two locations)
sed -i "s|FAIRFACE_BASE_DIR = .*|FAIRFACE_BASE_DIR = \"${FAIRFACE_PATH}\"|g" train_blip.py

# Update baseline_vlm_training.py
sed -i "s|base_dir = .*|base_dir = \"${YOUR_BASE_DIR}\"|" baseline_vlm_training.py

echo "Paths updated successfully!"
```

Run it:
```bash
chmod +x update_paths.sh
./update_paths.sh
```

#### Phase 4: Training

**Option A: Full Training Pipeline**

```bash
# 1. Train BLIP on FairFace 
python train_blip.py

# 2. Train ACMFO on COCO
python acmfo_training.py

# 3. Train baseline VLM
python baseline_vlm_training.py
```

**Option B: Quick Testing (reduced samples)**

Edit training configs to use fewer samples:

```python
# In acmfo_training.py, line 72-73:
TRAIN_SAMPLES = 1000  # Instead of 20000
VAL_SAMPLES = 200     # Instead of 5000

# In train_blip.py - subset the dataset
train_dataset = train_dataset[:1000]
val_dataset = val_dataset[:200]
```

Then run:
```bash
python acmfo_training.py  # 
```

#### Phase 5: Evaluation 

```bash
# 1. Comprehensive VLM evaluation
python evaluate_vlm.py

# 2. BLIP-specific evaluation
python evaluate_train_blip.py

# 3. Causal tracing analysis
python causal_tracing.py --model_path checkpoints/acmfo_coco_checkpoints/final_model
```

#### Phase 6: Analysis and Visualization

All visualization notebooks are included. To regenerate:

```bash
# Launch Jupyter
jupyter notebook

# Open and run:
# 1. coco_bias_analysis-3.ipynb
# 2. vlm_train-2.ipynb
# 3. baseline_vlm_training.ipynb
```

---

### Expected Results

After full reproduction, you should obtain:

**Fairness Metrics** (target ranges):
- DPD: 0.001-0.08 (lower is better)
- EOR: 0.9-1.1 (closer to 1.0 is better)
- BAS: -0.1 to 0.1 (closer to 0 is better)

**Performance Metrics** (competitive ranges):
- BLEU-4: 0.25-0.35
- ROUGE-L: 0.45-0.55

**Training Convergence**:
- Loss should decrease monotonically
- Validation loss should plateau after epoch 2-3
- No overfitting (train/val gap < 0.1)

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
BATCH_SIZE = 4  # Instead of 8 or 16

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

#### 2. Dataset Download Failures

**Error**: `403 Forbidden` or `Connection timeout`

**Solutions**:
```bash
# For COCO - use mirror
wget http://images.cocodataset.org/zips/train2017.zip
# If fails, try: https://pjreddie.com/projects/coco-mirror/

# For FairFace - use HuggingFace (automatic in train_blip.py)
# For Google Drive (CelebA) - use gdown:
pip install gdown
gdown --id FILE_ID -O output.zip
```

#### 3. NLTK Data Missing

**Error**: `LookupError: Resource punkt not found`

**Solution**:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

#### 4. Path Not Found Errors

**Error**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**:
```bash
# Check all paths are updated
grep -r "nfs/roberts" *.py  # Should show your updated paths

# Verify dataset structure
ls -R datasets/coco2017
ls -R datasets/fairface

# Fix permissions
chmod -R 755 datasets/
```

#### 5. CLIP Model Loading Issues

**Error**: `AttributeError: 'NoneType' object has no attribute 'to'`

**Solution**:
```python
# In evaluate_vlm.py, ensure CLIP loads successfully
try:
    self.clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True
    ).to(device)
    print("✓ CLIP loaded successfully")
except Exception as e:
    print(f"✗ CLIP loading failed: {e}")
    self.clip_model = None  # Graceful degradation
```

#### 6. Slow Training

**Issue**: Training is very slow

**Solutions**:
```python
# 1. Enable DataLoader workers
train_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE,
    num_workers=4,  # Use multiple processes
    pin_memory=True  # Faster GPU transfer
)

# 2. Use smaller subset for testing
TRAIN_SAMPLES = 1000  # Test with 1k samples first

# 3. Enable mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(...)
```

#### 7. Evaluation Metrics = 0

**Issue**: BLEU/ROUGE scores are 0 or very low

**Possible Causes**:
1. Model not trained enough
2. Caption tokenization mismatch
3. Empty predictions

**Debug**:
```python
# Check predictions
print("Generated:", generated_caption)
print("Reference:", reference_caption)

# Check tokenization
tokens = processor.tokenizer.tokenize(caption)
print("Tokens:", tokens)
```

---

### Performance Optimization Tips

1. **GPU Utilization**:
```python
# Monitor GPU usage
watch -n 1 nvidia-smi

# Maximize batch size without OOM
# Use gradient accumulation if needed
GRADIENT_ACCUMULATION_STEPS = 4
```

2. **Data Loading**:
```python
# Prefetch to GPU
for batch in prefetch_to_device(dataloader, device):
    ...

# Cache preprocessed data
cache_dir = "preprocessed_cache"
os.makedirs(cache_dir, exist_ok=True)
```

3. **Training Speed**:
```python
# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Use flash attention (if available)
model.config.use_flash_attention = True
```

---

## Citation

If you use this code or methodology, please cite:

```bibtex
@misc{vlm_fairness_2024,
  title={Fairness-Aware Cross-Modal Interpretability for Vision-Language Models},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/vlm-fairness}
}
```


## Support and Issues

For questions or issues:

1. Check this README thoroughly
2. Review error messages in console
3. Verify all paths are correctly updated
4. Check GPU memory with `nvidia-smi`
5. Ensure all datasets are downloaded correctly
6. Try with reduced sample sizes first

Common quick fixes:
```bash
# Reset everything
rm -rf checkpoints/ outputs/
python train_blip.py  # Start fresh

# Check versions
pip list | grep -E "torch|transformers|datasets"

# Update packages
pip install --upgrade transformers datasets torch
```
