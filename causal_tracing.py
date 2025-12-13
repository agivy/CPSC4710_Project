"""
Causal Tracing Analysis for BLIP Vision-Language Model

Implements causal intervention methods to identify:
1. Which visual patches are causally important for each generated token
2. Which model layers contain critical information
3. Comparison with attention weights (faithfulness analysis)

Based on: Meng et al. "Locating and Editing Factual Associations in GPT" (NeurIPS 2022)
Adapted for vision-language models with cross-modal reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from scipy.stats import spearmanr

# Set default font sizes
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# Configuration
MODEL_NAME = "Salesforce/blip-image-captioning-base"
OUTPUT_DIR = "vlm_trust_evaluation"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
print("="*80)
print("CAUSAL TRACING ANALYSIS FOR BLIP")
print("="*80)

# ============================================================================
# Load Model
# ============================================================================

print("\nLoading BLIP model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    use_safetensors=True
).to(device)
model.eval()
print(f"✓ Model loaded: {MODEL_NAME}")


# ============================================================================
# Causal Tracing Functions
# ============================================================================

class CausalTracer:
    """Implements causal tracing for BLIP model"""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device
        
    def get_clean_outputs(self, pixel_values):
        """Get clean (unperturbed) model outputs"""
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=pixel_values,
                max_length=50,
                num_beams=1,  # Greedy for consistency
                return_dict_in_generate=True,
                output_scores=True
            )
        return outputs
    
    def get_hidden_states(self, pixel_values, generated_ids):
        """Extract hidden states from all layers"""
        with torch.no_grad():
            # Forward pass through vision encoder
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
                output_hidden_states=True
            )
            
            # Get visual embeddings
            image_embeds = vision_outputs.last_hidden_state
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1], 
                dtype=torch.long, 
                device=self.device
            )
            
            # Forward through text decoder
            decoder_outputs = self.model.text_decoder(
                input_ids=generated_ids,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            
        return {
            'vision_hidden_states': vision_outputs.hidden_states,
            'decoder_hidden_states': decoder_outputs.hidden_states,
            'image_embeds': image_embeds
        }
    
    def corrupt_visual_patches(self, pixel_values, noise_level=0.5):
        """Corrupt visual input with noise"""
        noise = torch.randn_like(pixel_values) * noise_level
        return pixel_values + noise
    
    def restore_patch(self, corrupted_embeds, clean_embeds, patch_idx):
        """Restore a specific visual patch to clean state"""
        restored = corrupted_embeds.clone()
        restored[:, patch_idx, :] = clean_embeds[:, patch_idx, :]
        return restored
    
    def causal_trace_patches(self, pixel_values, num_samples=5):
        """
        Causal tracing: corrupt all patches, then restore one at a time
        Measure: change in model's output probability
        """
        # Get clean generation
        clean_outputs = self.get_clean_outputs(pixel_values)
        clean_ids = clean_outputs.sequences[0]
        clean_caption = self.processor.decode(clean_ids, skip_special_tokens=True)
        
        print(f"\nClean caption: {clean_caption}")
        
        # Get clean hidden states
        clean_states = self.get_hidden_states(pixel_values, clean_ids[:1].unsqueeze(0))
        clean_image_embeds = clean_states['image_embeds']
        
        num_patches = clean_image_embeds.shape[1]
        num_tokens = min(len(clean_ids) - 1, 15)  # Limit to first 15 tokens for speed
        
        # Initialize causal impact matrix
        causal_impact = np.zeros((num_tokens, num_patches))
        
        print(f"Tracing {num_tokens} tokens across {num_patches} patches...")
        
        for sample_idx in tqdm(range(num_samples), desc="Causal samples"):
            # Corrupt visual input
            corrupted_pixels = self.corrupt_visual_patches(pixel_values, noise_level=0.3)
            
            # Get corrupted embeddings
            with torch.no_grad():
                corrupted_vision = self.model.vision_model(
                    pixel_values=corrupted_pixels,
                    return_dict=True
                )
                corrupted_embeds = corrupted_vision.last_hidden_state
            
            # Get corrupted baseline probabilities for each token
            baseline_probs = []
            
            with torch.no_grad():
                # Autoregressive generation to get probabilities
                current_ids = clean_ids[:1].unsqueeze(0)  # Start token
                
                for t in range(num_tokens):
                    # Forward pass
                    outputs = self.model.text_decoder(
                        input_ids=current_ids,
                        encoder_hidden_states=corrupted_embeds,
                        return_dict=True
                    )
                    
                    # Get probability for the actual next token
                    next_token_id = clean_ids[t + 1].item()
                    logits = outputs.logits[0, -1]  # Last position
                    probs = F.softmax(logits, dim=-1)
                    prob = probs[next_token_id].item()
                    baseline_probs.append(prob)
                    
                    # Append actual token for next step
                    current_ids = torch.cat([
                        current_ids, 
                        clean_ids[t + 1].unsqueeze(0).unsqueeze(0)
                    ], dim=1)
            
            # Restore each patch and measure impact
            for patch_idx in range(num_patches):
                # Restore this patch
                restored_embeds = self.restore_patch(
                    corrupted_embeds, 
                    clean_image_embeds, 
                    patch_idx
                )
                
                # Get restored probabilities
                with torch.no_grad():
                    current_ids = clean_ids[:1].unsqueeze(0)
                    
                    for t in range(num_tokens):
                        outputs = self.model.text_decoder(
                            input_ids=current_ids,
                            encoder_hidden_states=restored_embeds,
                            return_dict=True
                        )
                        
                        next_token_id = clean_ids[t + 1].item()
                        logits = outputs.logits[0, -1]
                        probs = F.softmax(logits, dim=-1)
                        restored_prob = probs[next_token_id].item()
                        
                        # Causal impact = improvement from baseline
                        impact = restored_prob - baseline_probs[t]
                        causal_impact[t, patch_idx] += impact
                        
                        # Append actual token for next step
                        current_ids = torch.cat([
                            current_ids,
                            clean_ids[t + 1].unsqueeze(0).unsqueeze(0)
                        ], dim=1)
        
        # Average across samples
        causal_impact /= num_samples
        
        return {
            'causal_impact': causal_impact,
            'clean_caption': clean_caption,
            'clean_ids': clean_ids,
            'num_patches': num_patches,
            'num_tokens': num_tokens
        }
    
    def extract_attention_for_comparison(self, pixel_values, generated_ids):
        """Extract attention weights for faithfulness comparison"""
        try:
            with torch.no_grad():
                # Get visual embeddings
                vision_outputs = self.model.vision_model(
                    pixel_values=pixel_values,
                    return_dict=True
                )
                image_embeds = vision_outputs.last_hidden_state
                
                # Forward through decoder with attention output
                outputs = self.model.text_decoder(
                    input_ids=generated_ids[:, :1],
                    encoder_hidden_states=image_embeds,
                    return_dict=True,
                    output_attentions=True
                )
                
                # Try to get cross-attention
                if hasattr(outputs, 'cross_attentions') and outputs.cross_attentions:
                    # Average across heads and layers
                    cross_attn = torch.stack(outputs.cross_attentions).mean(dim=(0, 1))
                    return cross_attn.cpu().numpy()
                else:
                    return None
        except:
            return None


# ============================================================================
# Load Sample Images
# ============================================================================

def load_sample_images(num_images=3):
    """Load sample images from COCO"""
    from pycocotools.coco import COCO
    
    COCO_BASE_DIR = "/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/coco2017"
    annotation_file = os.path.join(COCO_BASE_DIR, 'annotations/captions_val2017.json')
    image_dir = os.path.join(COCO_BASE_DIR, 'val2017')
    
    coco = COCO(annotation_file)
    image_ids = list(coco.imgs.keys())
    
    # Sample random images
    np.random.seed(42)
    selected_ids = np.random.choice(image_ids, num_images, replace=False)
    
    images = []
    for img_id in selected_ids:
        img_info = coco.loadImgs(int(img_id))[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        images.append(image)
    
    return images


# ============================================================================
# Main Analysis
# ============================================================================

print("\nLoading sample images...")
sample_images = load_sample_images(num_images=3)
print(f"✓ Loaded {len(sample_images)} images")

tracer = CausalTracer(model, processor)

all_results = []

for img_idx, image in enumerate(sample_images):
    print(f"\n{'='*80}")
    print(f"ANALYZING IMAGE {img_idx + 1}/{len(sample_images)}")
    print(f"{'='*80}")
    
    # Process image
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs['pixel_values']
    
    # Run causal tracing
    results = tracer.causal_trace_patches(pixel_values, num_samples=5)
    
    # Extract attention for comparison
    attention = tracer.extract_attention_for_comparison(
        pixel_values, 
        results['clean_ids'][:1].unsqueeze(0)
    )
    
    results['attention'] = attention
    results['image'] = image
    results['image_idx'] = img_idx
    
    all_results.append(results)


# ============================================================================
# Visualization
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

def visualize_causal_tracing(results, output_path):
    """Create comprehensive causal tracing visualization"""
    
    causal_impact = results['causal_impact']
    caption = results['clean_caption']
    attention = results['attention']
    
    # Tokenize caption for display
    tokens = caption.split()
    num_tokens = min(len(tokens), causal_impact.shape[0])
    tokens = tokens[:num_tokens]
    
    # Create figure
    if attention is not None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        axes = [axes[0], axes[1], None]
    
    # 1. Causal Impact Heatmap
    ax = axes[0]
    im = ax.imshow(causal_impact[:num_tokens], 
                   aspect='auto', 
                   cmap='RdYlGn', 
                   interpolation='nearest',
                   vmin=-0.1, vmax=0.1)
    ax.set_xlabel('Visual Patch Index', fontweight='bold', fontsize=14)
    ax.set_ylabel('Token Position', fontweight='bold', fontsize=14)
    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels(tokens, fontsize=11)
    ax.set_title(f'Causal Impact: "{caption}"', fontweight='bold', fontsize=16, pad=15)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Causal Impact (Δ Probability)', fontweight='bold', fontsize=13)
    
    # 2. Top-k most important patches per token
    ax = axes[1]
    top_k = 10
    top_patches_per_token = []
    
    for t in range(num_tokens):
        top_indices = np.argsort(causal_impact[t])[-top_k:][::-1]
        top_patches_per_token.append(top_indices)
    
    # Plot as heatmap of top-k patches
    top_k_matrix = np.zeros((num_tokens, top_k))
    for t in range(num_tokens):
        top_k_matrix[t] = causal_impact[t, top_patches_per_token[t]]
    
    im2 = ax.imshow(top_k_matrix, 
                    aspect='auto', 
                    cmap='RdYlGn',
                    interpolation='nearest',
                    vmin=-0.1, vmax=0.1)
    ax.set_xlabel(f'Top-{top_k} Most Important Patches (Ranked)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Token Position', fontweight='bold', fontsize=14)
    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels(tokens, fontsize=11)
    ax.set_title(f'Top-{top_k} Causally Important Patches per Token', 
                 fontweight='bold', fontsize=16, pad=15)
    
    cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label('Causal Impact', fontweight='bold', fontsize=13)
    
    # 3. Attention vs Causal Impact comparison (if available)
    if attention is not None and axes[2] is not None:
        ax = axes[2]
        
        # Flatten both for correlation
        causal_flat = causal_impact[:num_tokens].flatten()
        attn_flat = attention[:num_tokens, :causal_impact.shape[1]].flatten()
        
        # Compute correlation
        if len(causal_flat) == len(attn_flat):
            correlation, p_value = spearmanr(causal_flat, attn_flat)
            
            # Scatter plot
            ax.scatter(attn_flat, causal_flat, alpha=0.3, s=20)
            ax.set_xlabel('Attention Weight', fontweight='bold', fontsize=14)
            ax.set_ylabel('Causal Impact', fontweight='bold', fontsize=14)
            ax.set_title(f'Faithfulness: Attention vs Causal Impact\n' + 
                        f'Spearman ρ = {correlation:.3f} (p < {p_value:.2e})',
                        fontweight='bold', fontsize=16, pad=15)
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(attn_flat, causal_flat, 1)
            p = np.poly1d(z)
            x_line = np.linspace(attn_flat.min(), attn_flat.max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Linear fit')
            ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# Generate visualizations for each image
for results in all_results:
    img_idx = results['image_idx']
    output_path = os.path.join(OUTPUT_DIR, f'causal_tracing_image_{img_idx + 1}.pdf')
    visualize_causal_tracing(results, output_path)


# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "="*80)
print("CAUSAL TRACING SUMMARY")
print("="*80)

for results in all_results:
    img_idx = results['image_idx']
    caption = results['clean_caption']
    causal_impact = results['causal_impact']
    attention = results['attention']
    
    print(f"\nImage {img_idx + 1}:")
    print(f"  Caption: {caption}")
    print(f"  Causal impact range: [{causal_impact.min():.4f}, {causal_impact.max():.4f}]")
    print(f"  Mean causal impact: {causal_impact.mean():.4f}")
    print(f"  Std causal impact: {causal_impact.std():.4f}")
    
    # Top-5 most important patches overall
    top_5_patches = np.argsort(causal_impact.sum(axis=0))[-5:][::-1]
    print(f"  Top-5 most important patches: {top_5_patches}")
    
    # Faithfulness (if attention available)
    if attention is not None:
        num_tokens = min(causal_impact.shape[0], attention.shape[0])
        causal_flat = causal_impact[:num_tokens].flatten()
        attn_flat = attention[:num_tokens, :causal_impact.shape[1]].flatten()
        
        if len(causal_flat) == len(attn_flat):
            correlation, p_value = spearmanr(causal_flat, attn_flat)
            print(f"  Attention faithfulness (Spearman ρ): {correlation:.3f} (p={p_value:.2e})")
            
            if correlation < 0.3:
                print(f"    → LOW faithfulness: Attention does NOT reflect causal importance!")
            elif correlation < 0.6:
                print(f"    → MODERATE faithfulness")
            else:
                print(f"    → HIGH faithfulness: Attention reflects causal importance")


# ============================================================================
# Create Summary Comparison Figure
# ============================================================================

print("\n" + "="*80)
print("CREATING SUMMARY COMPARISON")
print("="*80)

fig, axes = plt.subplots(len(all_results), 2, figsize=(16, 6*len(all_results)))
if len(all_results) == 1:
    axes = axes.reshape(1, -1)

for idx, results in enumerate(all_results):
    causal_impact = results['causal_impact']
    caption = results['clean_caption']
    
    # Get actual number of tokens from causal_impact shape
    num_tokens_actual = causal_impact.shape[0]
    all_tokens = caption.split()
    
    # CRITICAL: Only use as many tokens as we have causal impact data for
    tokens = all_tokens[:num_tokens_actual]
    
    # If we have more causal impact rows than tokens, truncate causal impact
    if num_tokens_actual > len(all_tokens):
        causal_impact = causal_impact[:len(all_tokens)]
        num_tokens_actual = len(all_tokens)
    
    # Left: Average causal impact per patch
    ax = axes[idx, 0]
    avg_impact = causal_impact.mean(axis=0)
    ax.bar(range(len(avg_impact)), avg_impact, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Visual Patch Index', fontweight='bold', fontsize=14)
    ax.set_ylabel('Average Causal Impact', fontweight='bold', fontsize=14)
    ax.set_title(f'Image {idx+1}: Patch Importance\n"{caption}"', 
                 fontweight='bold', fontsize=15)
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    # Right: Average causal impact per token
    ax = axes[idx, 1]
    avg_impact_per_token = causal_impact.mean(axis=1)
    
    # Double-check lengths match
    assert len(avg_impact_per_token) == len(tokens), f"Mismatch: {len(avg_impact_per_token)} vs {len(tokens)}"
    
    ax.barh(range(len(tokens)), avg_impact_per_token, color='coral', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=12)
    ax.set_xlabel('Average Causal Impact', fontweight='bold', fontsize=14)
    ax.set_ylabel('Token', fontweight='bold', fontsize=14)
    ax.set_title(f'Image {idx+1}: Token Sensitivity', fontweight='bold', fontsize=15)
    ax.grid(True, axis='x', alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.invert_yaxis()

plt.tight_layout()
summary_path = os.path.join(OUTPUT_DIR, 'causal_tracing_summary.pdf')
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved summary: {summary_path}")
plt.close()

print("\n" + "="*80)
print("CAUSAL TRACING COMPLETE!")
print("="*80)
print(f"\nGenerated files in {OUTPUT_DIR}/:")
print(f"  - causal_tracing_image_1.pdf (detailed analysis)")
print(f"  - causal_tracing_image_2.pdf")
print(f"  - causal_tracing_image_3.pdf")
print(f"  - causal_tracing_summary.pdf (comparison across images)")
print("\nKey Insights:")
print("  1. Causal impact shows which patches ACTUALLY matter for generation")
print("  2. Comparison with attention reveals faithfulness (or lack thereof)")
print("  3. Low correlation = attention is NOT causally faithful!")
print("="*80)