# Fairness-Aware Cross-Modal Interpretability for Vision-Language Models

This repository contains the implementation and evaluation code for training and assessing Vision-Language Models (VLMs) with a focus on fairness, interpretability, and trustworthiness.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset Setup](#dataset-setup)
- [Training Scripts](#training-scripts)
- [Evaluation Scripts](#evaluation-scripts)
- [Results and Analysis](#results-and-analysis)
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

```
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
│
├── TRAINING SCRIPTS
├── acmfo_training.py                           # ACMFO method on COCO dataset
├── baseline_vlm_training.py                    # Baseline VLM fine-tuning
├── train_blip.py                               # Standard BLIP training on FairFace
│
├── DATASET DOWNLOAD SCRIPTS
├── download_coco.sh                            # MS-COCO 2017 dataset download
├── download_fairface.sh                        # FairFace dataset download (bash)
├── download_fairface.py                        # FairFace dataset download (Python)
├── download_celeba.sh                          # CelebA dataset download
│
├── EVALUATION SCRIPTS
├── evaluate_vlm.py                             # Comprehensive trustworthiness evaluation
├── evaluate_train_blip.py                      # BLIP model evaluation
├── causal_tracing.py                           # Causal tracing for interpretability
│
├── NOTEBOOKS
├── vlm_train-2.ipynb                           # Main training notebook
├── baseline_vlm_training.ipynb                 # Baseline training notebook
├── coco_bias_analysis-3.ipynb                  # COCO bias analysis
│
├── RESULTS AND VISUALIZATIONS
├── fairness_evaluation_results.csv             # Fairness metrics
├── fairness_summary.txt                        # Summary of fairness evaluation
├── fairness_summary.json                       # JSON format fairness summary
├── final_evaluation_results.csv                # Complete evaluation results
├── complete_evaluation_results.csv             # Detailed evaluation
├── blip_evaluation_report.txt                  # BLIP evaluation report
├── blip_coco_results.csv                       # BLIP results on COCO
├── blip_coco_multi_reference_scores.csv        # Multi-reference BLEU scores
├── blip2_captions.csv                          # Generated captions
├── coco_bias_report.txt                        # COCO bias analysis report
├── coco_bias_report.json                       # COCO bias analysis (JSON)
│
└── PLOTS AND VISUALIZATIONS
    ├── fairness_evaluation_plots.png           # Fairness metric plots
    ├── gender_distribution.png                 # Gender distribution analysis
    ├── people_distribution.png                 # People count distribution
    ├── activity_gender_bias.png                # Activity-gender bias (original)
    ├── activity_gender_bias_corrected.png      # Activity-gender bias (corrected)
    ├── object_gender_bias.png                  # Object-gender bias (original)
    ├── object_gender_bias_corrected.png        # Object-gender bias (corrected)
    ├── word_gender_bias_corrected.png          # Word-gender bias analysis
    ├── caption_length_distribution.png         # Caption length statistics
    ├── sample_images_captions.png              # Sample outputs
    ├── flickr30k_gender_distribution.png       # Flickr30k gender analysis
    ├── flickr30k_activity_bias.png             # Flickr30k activity bias
    ├── flickr30k_context_bias.png              # Flickr30k context bias
    ├── flickr30k_caption_quality.png           # Flickr30k caption quality
    └── coco_comprehensive_bias_analysis.{png,pdf}  # Complete COCO bias analysis
```

---

## Requirements

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

## Training Scripts

### 1. ACMFO Training (Main Method)

**File**: `acmfo_training.py`

**Description**: Implements the Adaptive Cross-Modal Fairness Optimizer with multi-objective loss on MS-COCO.

**Key Features**:
- Demographic parity via KL divergence regularization
- Cross-modal disentanglement via mutual information minimization
- CLIP-based gender inference (no demographic labels needed)
- k-means clustering for caption space approximation

**Configuration** (Lines 54-87):
```python
class ACMFOConfig:
    # Paths - UPDATE THESE!
    COCO_BASE_DIR = "/your/path/to/coco2017"
    OUTPUT_DIR = "acmfo_coco_checkpoints"
    
    # Model
    BASE_MODEL = "Salesforce/blip-image-captioning-base"
    
    # Training
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 5e-5
    TRAIN_SAMPLES = 20000  # Subset for faster training
    VAL_SAMPLES = 5000
    
    # ACMFO hyperparameters
    LAMBDA_FAIRNESS = 1.0       # Fairness loss weight
    LAMBDA_CROSS_MODAL = 0.05   # Cross-modal loss weight
    
    # Fairness parameters
    NUM_CAPTION_CLUSTERS = 100
    GENDER_CONFIDENCE_THRESHOLD = 0.3
```

**Usage**:
```bash
# 1. Update COCO_BASE_DIR in line 58
nano acmfo_training.py

# 2. Run training
python acmfo_training.py
```

**Outputs**:
```
acmfo_coco_checkpoints/
├── checkpoint_epoch_1/
├── checkpoint_epoch_2/
├── checkpoint_epoch_3/
├── training_log.json
├── fairness_metrics.json
└── final_model/
```

---

### 2. Baseline VLM Training

**File**: `baseline_vlm_training.py`

**Description**: Fine-tunes GIT (microsoft/git-base) on FairFace with demographic annotations.

**Configuration** (Lines 49-100):
```python
class Config:
    # Model
    model_name = "microsoft/git-base"  # 0.6B params
    
    # Data
    fairface_dataset = "HuggingFaceM4/FairFace"
    img_size = 224
    max_caption_length = 50
    
    # Training
    batch_size = 16
    num_epochs = 4
    learning_rate = 5e-5
    
    # Paths - UPDATE THESE!
    base_dir = "/your/path/to/project"
    output_dir = f"{base_dir}/baseline_vlm"
```

**Usage**:
```bash
# 1. Update base_dir in line 69
nano baseline_vlm_training.py

# 2. Run training
python baseline_vlm_training.py
```

**Outputs**:
```
baseline_vlm/
├── checkpoints/
├── plots/
│   ├── training_curves.png
│   └── demographic_distribution.png
└── results/
    └── evaluation_metrics.json
```

---

### 3. Standard BLIP Training on FairFace

**File**: `train_blip.py`

**Description**: Fine-tunes BLIP on FairFace with demographic-aware caption generation.

**Key Features**:
- Automatic FairFace download from Hugging Face
- Caption generation using all 3 attributes (age, gender, race)
- Training with demographic supervision

**Configuration** (Lines 243-282):
```python
# Training hyperparameters
BATCH_SIZE = 32  # FairFace images are small
EPOCHS = 5
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0
```

**Usage**:
```bash
# 1. Update FAIRFACE_BASE_DIR in lines 31 and 92
nano train_blip.py

# 2. Run training (automatically downloads FairFace)
python train_blip.py
```


**Outputs**:
```
checkpoints/
├── blip-finetuned-fairface/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── preprocessor_config.json
└── training_history.json
```

**Caption Generation Strategy**:
The script generates captions using all three FairFace attributes:
```python
# Example: "a photo of a young adult Asian woman"
caption = f"a photo of a {age_desc} {race} {gender_noun}"
```

Age mapping:
- 0-2: infant
- 3-9: child
- 10-19: teenager
- 20-29: young adult
- 30-39: adult
- 40-49, 50-59: middle-aged adult
- 60-69: senior
- 70+: elderly person

---

## Evaluation Scripts

### 1. Comprehensive VLM Evaluation

**File**: `evaluate_vlm.py`

**Description**: Evaluates pretrained VLMs on four trustworthiness dimensions.

**Evaluation Metrics**:

1. **Performance**:
   - BLEU-1, BLEU-2, BLEU-3, BLEU-4
   - ROUGE-L

2. **Fairness**:
   - Demographic Parity Difference (DPD)
   - Equalized Odds Ratio (EOR)
   - Bias Amplification Score (BAS)

3. **Interpretability**:
   - Attention faithfulness
   - Gradient-based importance

4. **Reliability**:
   - Calibration error
   - Out-of-distribution (OOD) confidence

**Configuration** (Lines 43-49):
```python
COCO_BASE_DIR = "/your/path/to/coco2017"  # UPDATE THIS!
OUTPUT_DIR = "vlm_trust_evaluation"
BATCH_SIZE = 16
MAX_SAMPLES = None  # Use full validation set
NUM_VISUALIZATION_SAMPLES = 10
```

**Models Evaluated**:
```python
MODELS = {
    "BLIP-Base": {
        "model_class": BlipForConditionalGeneration,
        "model_name": "Salesforce/blip-image-captioning-base",
        "processor_class": BlipProcessor,
        "size_m": 223,  # ~223M parameters
    },
}
```

**Usage**:
```bash
# 1. Update COCO_BASE_DIR in line 44
nano evaluate_vlm.py

# 2. Run evaluation
python evaluate_vlm.py
```

**Outputs**:
```
vlm_trust_evaluation/
├── results/
│   ├── fairness_metrics.csv
│   ├── performance_metrics.csv
│   ├── interpretability_scores.csv
│   └── reliability_metrics.csv
├── plots/
│   ├── bias_analysis.png
│   ├── attention_maps.png
│   └── calibration_curves.png
└── evaluation_report.txt
```

**Key Fairness Metrics Explained**:

- **DPD (Demographic Parity Difference)**: 
  - Measures difference in positive outcome rates between groups
  - Target: ≤ 0.08
  - Formula: |P(Ŷ=1|G=male) - P(Ŷ=1|G=female)|

- **EOR (Equalized Odds Ratio)**:
  - Measures TPR ratio between groups
  - Target: [0.9, 1.1]
  - Formula: TPR(male) / TPR(female)

- **BAS (Bias Amplification Score)**:
  - Compares model bias to dataset bias
  - Target: |BAS| < 0.1
  - Formula: (model_bias - dataset_bias) / dataset_bias

---

### 2. BLIP Model Evaluation

**File**: `evaluate_train_blip.py`

**Description**: Evaluates fine-tuned BLIP models with fairness analysis.

**Usage**:
```bash
python evaluate_train_blip.py \
    --model_path checkpoints/blip-finetuned-fairface \
    --dataset_path /path/to/fairface \
    --output_dir evaluation_results
```

**Outputs**:
- BLEU and ROUGE scores
- Demographic-stratified performance
- Bias amplification metrics
- Confusion matrices

---

### 3. Causal Tracing

**File**: `causal_tracing.py`

**Description**: Implements causal tracing for interpretability analysis.

**Key Features**:
- Layer-wise importance analysis
- Attention head attribution
- Cross-modal influence tracking

**Usage**:
```bash
python causal_tracing.py \
    --model_path checkpoints/acmfo_coco_checkpoints/final_model \
    --image_path sample_images/test_image.jpg
```

---

## Results and Analysis

### Evaluation Results

The repository includes several pre-computed evaluation results:

#### 1. Fairness Summary (`fairness_summary.txt`)
```
FAIRNESS EVALUATION SUMMARY
======================================================================
DPD: 0.0023 (target ≤ 0.08): PASS
EOR: inf (target [0.9, 1.1]): FAIL
BAS: 1.0000 (target |BAS| < 0.1): FAIL
BLEU-4: 0.0351
ROUGE-L: 0.2321
```

**Interpretation**:
- DPD passes (0.0023 << 0.08): Good demographic parity
- EOR fails (inf): Need more samples with true positives
- BAS fails (1.0): Significant bias amplification
- Performance: BLEU-4=0.0351, ROUGE-L=0.2321

#### 2. Complete Evaluation Results

Available in CSV format:
- `fairness_evaluation_results.csv` - Per-group fairness metrics
- `final_evaluation_results.csv` - Aggregated results
- `complete_evaluation_results.csv` - Detailed per-sample results
- `blip_coco_results.csv` - BLIP performance on COCO
- `blip_coco_multi_reference_scores.csv` - Multi-reference BLEU

#### 3. Bias Analysis

**COCO Bias Analysis** (`coco_bias_report.json`):
```json
{
  "gender_distribution": {
    "male": 0.45,
    "female": 0.35,
    "unknown": 0.20
  },
  "activity_bias": {
    "male": ["sports", "working", "playing"],
    "female": ["cooking", "caring", "shopping"]
  },
  "object_association": {
    "male": ["car", "computer", "bike"],
    "female": ["kitchen", "children", "flowers"]
  }
}
```

### Visualization Gallery

The repository includes numerous pre-generated plots:

1. **Fairness Plots**:
   - `fairness_evaluation_plots.png` - Multi-metric fairness dashboard
   - `gender_distribution.png` - Gender distribution in dataset
   - `people_distribution.png` - Number of people per image

2. **Bias Analysis**:
   - `activity_gender_bias.png` / `activity_gender_bias_corrected.png`
   - `object_gender_bias.png` / `object_gender_bias_corrected.png`
   - `word_gender_bias_corrected.png`

3. **Dataset Analysis**:
   - `flickr30k_gender_distribution.png`
   - `flickr30k_activity_bias.png`
   - `flickr30k_context_bias.png`
   - `flickr30k_caption_quality.png`

4. **COCO Analysis**:
   - `coco_comprehensive_bias_analysis.png` - Complete bias analysis
   - `coco_comprehensive_bias_analysis.pdf` - High-res PDF version

5. **Model Outputs**:
   - `sample_images_captions.png` - Generated caption examples
   - `caption_length_distribution.png` - Caption length statistics

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

# 3. FairFace will be downloaded automatically by train_blip.py
# Or manually:
nano download_fairface.sh  # Update BASE_DIR
chmod +x download_fairface.sh
./download_fairface.sh


# 4. CelebA (optional)
nano download_celeba.sh  # Update BASE_DIR
chmod +x download_celeba.sh
./download_celeba.sh
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
python acmfo_training.py  
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

# Update packages
pip install --upgrade transformers datasets torch
```

---

## License

This project is released under the MIT License. See `LICENSE` file for details.

---
