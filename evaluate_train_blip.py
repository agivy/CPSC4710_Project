"""
Evaluation script for fine-tuned BLIP model on FairFace dataset
- Loads best checkpoint
- Generates captions for sample images
- Computes BLEU and ROUGE-L scores
- Visualizes results and training history
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
import pandas as pd
from datasets import load_dataset

# BLEU and ROUGE metrics
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuration
CHECKPOINT_PATH = "checkpoints/blip-finetuned-fairface"
FAIRFACE_BASE_DIR = "/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/fairface"
OUTPUT_DIR = "evaluation_results"
BATCH_SIZE = 32
NUM_SAMPLE_IMAGES = 20

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ============================================================================
# Dataset Class
# ============================================================================

class FairFaceHFDataset(Dataset):
    """FairFace Dataset from Hugging Face with caption generation"""
    
    def __init__(self, hf_dataset, processor, max_length=50, mode='eval'):
        self.dataset = hf_dataset
        self.processor = processor
        self.max_length = max_length
        self.mode = mode
        
        # Print statistics
        genders = [item['gender'] for item in self.dataset]
        races = [item['race'] for item in self.dataset]
        ages = [item['age'] for item in self.dataset]
        
        print(f"\n{mode.upper()} Dataset Statistics:")
        print(f"  Total samples: {len(self.dataset):,}")
        
        print(f"\n  Gender distribution:")
        for gender, count in Counter(genders).items():
            print(f"    {gender}: {count:,} ({count/len(self.dataset)*100:.1f}%)")
        
        print(f"\n  Race distribution:")
        for race, count in Counter(races).most_common():
            print(f"    {race}: {count:,} ({count/len(self.dataset)*100:.1f}%)")
        
        print(f"\n  Age distribution:")
        for age, count in sorted(Counter(ages).items()):
            print(f"    {age}: {count:,} ({count/len(self.dataset)*100:.1f}%)")
    
    def generate_caption(self, age, gender, race):
        """Generate ground truth caption using FairFace attributes"""
        
        age_map = {
            '0-2': 'infant', '3-9': 'child', '10-19': 'teenager',
            '20-29': 'young adult', '30-39': 'adult',
            '40-49': 'middle-aged adult', '50-59': 'middle-aged adult',
            '60-69': 'senior', 'more than 70': 'elderly person'
        }
        
        age_desc = age_map.get(age, 'adult')
        gender_noun = 'man' if gender == 'Male' else 'woman'
        
        # Return the full caption (matching training)
        caption = f"a photo of a {age_desc} {race} {gender_noun}"
        
        return caption
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get image (already a PIL Image from HF dataset)
        image = item['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get attributes
        age = item['age']
        gender = item['gender']
        race = item['race']
        
        # Generate ground truth caption
        caption = self.generate_caption(age, gender, race)
        
        # Process image
        encoding = self.processor(
            images=image,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'image': image,  # Keep original for visualization
            'caption': caption,
            'gender': gender,
            'race': race,
            'age': age
        }


# ============================================================================
# Load Model and Data
# ============================================================================

print("\n" + "="*80)
print("LOADING MODEL AND DATA")
print("="*80)

# Load processor and model
print(f"\nLoading model from: {CHECKPOINT_PATH}")
processor = BlipProcessor.from_pretrained(CHECKPOINT_PATH)
model = BlipForConditionalGeneration.from_pretrained(CHECKPOINT_PATH).to(device)
model.eval()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Load FairFace dataset
print(f"\nLoading FairFace dataset from: {FAIRFACE_BASE_DIR}")
dataset = load_dataset("HuggingFaceM4/FairFace", "0.25", cache_dir=FAIRFACE_BASE_DIR)

val_hf = dataset['validation']
print(f"Validation samples: {len(val_hf):,}")

# Create dataset
val_dataset = FairFaceHFDataset(val_hf, processor, mode='eval')

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ============================================================================
# Generate Captions for Sample Images
# ============================================================================

print("\n" + "="*80)
print("GENERATING CAPTIONS FOR SAMPLE IMAGES")
print("="*80)

# Select random samples for visualization
np.random.seed(42)
sample_indices = np.random.choice(len(val_dataset), NUM_SAMPLE_IMAGES, replace=False)

sample_images = []
ground_truth_captions = []
generated_captions = []

print(f"\nGenerating captions for {NUM_SAMPLE_IMAGES} sample images...")

for idx in tqdm(sample_indices, desc="Generating samples"):
    # Convert numpy int64 to Python int for HF dataset compatibility
    idx = int(idx)
    sample = val_dataset[idx]
    
    # Get image and ground truth
    image = sample['image']
    gt_caption = sample['caption']
    
    # Generate caption
    pixel_values = sample['pixel_values'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=50,
            num_beams=3,
            early_stopping=True
        )
    
    gen_caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    sample_images.append(image)
    ground_truth_captions.append(gt_caption)
    generated_captions.append(gen_caption)
    
    # Print first few examples
    if len(sample_images) <= 5:
        print(f"\nExample {len(sample_images)}:")
        print(f"  Ground Truth: {gt_caption}")
        print(f"  Generated:    {gen_caption}")


# ============================================================================
# Visualize Sample Results
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATION")
print("="*80)

# Create figure with subplots
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for idx in range(NUM_SAMPLE_IMAGES):
    ax = axes[idx]
    
    # Display image
    ax.imshow(sample_images[idx])
    ax.axis('off')
    
    # Add captions as title
    gt = ground_truth_captions[idx]
    gen = generated_captions[idx]
    
    # Wrap text if too long
    max_chars = 40
    if len(gt) > max_chars:
        gt = gt[:max_chars-3] + '...'
    if len(gen) > max_chars:
        gen = gen[:max_chars-3] + '...'
    
    title = f"GT: {gt}\nGen: {gen}"
    ax.set_title(title, fontsize=9, pad=10, color='black', 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='GT = Ground Truth (Blue)',
           markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Gen = Generated (follows GT text)',
           markerfacecolor='red', markersize=10)
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12, 
           bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_path = os.path.join(OUTPUT_DIR, 'sample_captions.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSample captions visualization saved to: {output_path}")
plt.close()


# ============================================================================
# Evaluate BLEU and ROUGE Scores
# ============================================================================

print("\n" + "="*80)
print("COMPUTING BLEU AND ROUGE SCORES")
print("="*80)

all_references = []
all_hypotheses = []

print("\nGenerating captions for entire validation set...")

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        pixel_values = batch['pixel_values'].to(device)
        gt_captions = batch['caption']
        
        # Generate captions
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=50,
            num_beams=3,
            early_stopping=True
        )
        
        # Decode
        gen_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Store for metrics
        for gt, gen in zip(gt_captions, gen_captions):
            # BLEU expects tokenized references as list of lists
            ref_tokens = gt.lower().split()
            hyp_tokens = gen.lower().split()
            
            all_references.append([ref_tokens])  # List of references for each hypothesis
            all_hypotheses.append(hyp_tokens)

print(f"\nTotal captions evaluated: {len(all_hypotheses):,}")

# Compute BLEU scores (1-4)
smoothing = SmoothingFunction()

print("\nComputing BLEU scores...")
bleu_scores = {}

for n in range(1, 5):
    weights = tuple([1.0/n] * n + [0.0] * (4-n))
    
    bleu_score = corpus_bleu(
        all_references,
        all_hypotheses,
        weights=weights,
        smoothing_function=smoothing.method1
    )
    
    bleu_scores[f'BLEU-{n}'] = bleu_score
    print(f"  BLEU-{n}: {bleu_score:.4f}")

# Compute ROUGE-L
print("\nComputing ROUGE-L score...")
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

rouge_scores = []
for ref_tokens, hyp_tokens in tqdm(zip(all_references, all_hypotheses), 
                                    total=len(all_references),
                                    desc="Computing ROUGE"):
    ref_text = ' '.join(ref_tokens[0])
    hyp_text = ' '.join(hyp_tokens)
    
    score = scorer.score(ref_text, hyp_text)
    rouge_scores.append(score['rougeL'].fmeasure)

rouge_l_score = np.mean(rouge_scores)
print(f"  ROUGE-L: {rouge_l_score:.4f}")

# Save metrics
metrics = {
    **bleu_scores,
    'ROUGE-L': rouge_l_score
}

metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved to: {metrics_path}")

# Print summary
print("\n" + "="*80)
print("EVALUATION METRICS SUMMARY")
print("="*80)
for metric, score in metrics.items():
    print(f"  {metric}: {score:.4f}")


# ============================================================================
# Plot Training History
# ============================================================================

print("\n" + "="*80)
print("PLOTTING TRAINING HISTORY")
print("="*80)

history_path = 'checkpoints/training_history.json'

if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(train_loss) + 1)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    plt.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add min val loss marker
    min_val_idx = np.argmin(val_loss)
    plt.plot(epochs[min_val_idx], val_loss[min_val_idx], 'g*', 
             markersize=20, label=f'Best Val Loss: {val_loss[min_val_idx]:.4f}')
    plt.legend(fontsize=12, loc='upper right')
    
    # Formatting
    plt.xticks(epochs, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save
    loss_plot_path = os.path.join(OUTPUT_DIR, 'training_loss.pdf')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining loss plot saved to: {loss_plot_path}")
    plt.close()
    
    print(f"\nTraining Summary:")
    print(f"  Total Epochs: {len(epochs)}")
    print(f"  Final Training Loss: {train_loss[-1]:.4f}")
    print(f"  Final Validation Loss: {val_loss[-1]:.4f}")
    print(f"  Best Validation Loss: {val_loss[min_val_idx]:.4f} (Epoch {min_val_idx+1})")
else:
    print(f"\nWarning: Training history not found at {history_path}")


# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)

print(f"\nAll results saved to: {OUTPUT_DIR}/")
print(f"\nGenerated files:")
print(f"  1. {OUTPUT_DIR}/sample_captions.pdf - Sample images with captions")
print(f"  2. {OUTPUT_DIR}/training_loss.pdf - Training history plot")
print(f"  3. {OUTPUT_DIR}/metrics.json - BLEU and ROUGE scores")

print("\n" + "="*80)
print("DONE!")
print("="*80)