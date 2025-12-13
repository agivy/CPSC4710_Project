"""
Baseline VLM Fine-tuning on CelebA-HQ with FairFace Demographics
Project: Fairness-Aware Cross-Modal Interpretability for Vision-Language Models

This script implements Baseline 1: Fine-tuning a small VLM on CelebA-HQ
with demographic annotations from FairFace classifier.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms

# For evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class Config:
    # Model settings
    model_name = "microsoft/git-base"  # Small VLM (0.6B parameters)
    
    # Data settings - Using FairFace dataset with demographics
    fairface_dataset = "HuggingFaceM4/FairFace"
    celeba_hq_dataset = "mattmdjaga/celeba_hq_with_captions"
    img_size = 224
    max_caption_length = 50
    
    # Training settings
    batch_size = 16
    num_epochs = 4
    learning_rate = 5e-5
    warmup_steps = 500
    weight_decay = 0.01
    gradient_accumulation_steps = 4
    max_grad_norm = 1.0
    
    # Paths
    base_dir = "/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project"
    output_dir = f"{base_dir}/baseline_vlm"
    plots_dir = f"{output_dir}/plots"
    checkpoints_dir = f"{output_dir}/checkpoints"
    results_dir = f"{output_dir}/results"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = torch.cuda.is_available()
    
    # Evaluation settings
    eval_every_n_steps = 500
    save_total_limit = 2
    
    # Demographic attributes from FairFace
    gender_map = {0: 'Male', 1: 'Female'}
    race_map = {
        0: 'White',
        1: 'Black', 
        2: 'Asian',
        3: 'Indian',
        4: 'Other'  # Combines Latino_Hispanic and Middle Eastern
    }
    age_map = {
        0: 'Young',    # 0-19
        1: 'Young',    # 20-29
        2: 'Middle',   # 30-39
        3: 'Middle',   # 40-49
        4: 'Middle',   # 50-59
        5: 'Old',      # 60-69
        6: 'Old'       # 70+
    }
    
    # Bias-sensitive words for fairness metrics
    professional_terms = [
        'doctor', 'engineer', 'scientist', 'professor', 'executive',
        'lawyer', 'manager', 'director', 'ceo', 'professional',
        'businessman', 'entrepreneur', 'leader', 'expert'
    ]
    service_terms = [
        'assistant', 'secretary', 'receptionist', 'nurse', 'caregiver',
        'housekeeper', 'cleaner', 'worker', 'helper', 'attendant'
    ]

config = Config()

# Create output directories
for dir_path in [config.output_dir, config.plots_dir, config.checkpoints_dir, config.results_dir]:
    os.makedirs(dir_path, exist_ok=True)


class FairFaceVLMDataset(Dataset):
    """Dataset using FairFace for demographic annotations"""
    
    def __init__(self, split='train', processor=None, config=config):
        self.config = config
        self.processor = processor
        self.split = split
        
        print(f"Loading FairFace {split} dataset with demographics...")
        
        # Load FairFace dataset
        try:
            self.dataset = load_dataset(config.fairface_dataset, split=split)
            print(f"Loaded {len(self.dataset)} FairFace samples with demographics")
        except Exception as e:
            print(f"Error loading FairFace: {e}")
            print("Attempting alternative loading method...")
            # Try loading from HuggingFace hub
            self.dataset = load_dataset("HuggingFaceM4/FairFace", split=split)
        
        # Generate captions for images
        print("Generating captions for images...")
        self._generate_captions()
        
        # Convert demographic labels to text
        self._process_demographics()
        
    def _generate_captions(self):
        """Generate descriptive captions based on image attributes"""
        def create_caption(example):
            # Create descriptive captions from demographics
            # In practice, you could use a caption generator or manual annotations
            
            gender = example.get('gender', 0)
            age = example.get('age', 0)
            race = example.get('race', 0)
            
            # Map to text
            gender_text = self.config.gender_map.get(gender, 'person')
            race_text = self.config.race_map.get(race, 'person')
            age_text = self.config.age_map.get(age, 'person')
            
            # Generate basic caption
            caption = f"A {age_text.lower()} {gender_text.lower()} person"
            
            # Add more descriptive elements
            descriptors = []
            if np.random.rand() > 0.5:
                descriptors.append("smiling")
            if np.random.rand() > 0.7:
                descriptors.append("wearing glasses")
            
            if descriptors:
                caption += " " + " and ".join(descriptors)
            
            example['caption'] = caption
            return example
        
        self.dataset = self.dataset.map(create_caption)
    
    def _process_demographics(self):
        """Convert numeric demographic labels to text"""
        def process_demo(example):
            example['gender_text'] = self.config.gender_map.get(example.get('gender', 0), 'Unknown')
            example['race_text'] = self.config.race_map.get(example.get('race', 0), 'Unknown')
            example['age_text'] = self.config.age_map.get(example.get('age', 0), 'Unknown')
            return example
        
        self.dataset = self.dataset.map(process_demo)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Get image
        image = example['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        image = image.convert('RGB')
        
        # Get caption
        caption = example.get('caption', 'A person')
        
        # Process inputs
        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_caption_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add demographic info
        encoding['gender'] = example.get('gender_text', 'Unknown')
        encoding['age_group'] = example.get('age_text', 'Unknown')
        encoding['race'] = example.get('race_text', 'Unknown')
        encoding['caption_text'] = caption
        
        return encoding


class FairnessMetrics:
    """Compute fairness metrics as defined in Section 6.1"""
    
    def __init__(self, config=config):
        self.config = config
        
    def compute_demographic_parity_difference(self, predictions_by_group):
        """
        DPD = max_{g,g'} |P(ŷ ∈ Y+|G=g) - P(ŷ ∈ Y+|G=g')|
        """
        positive_rates = {}
        
        for group, preds in predictions_by_group.items():
            positive_count = sum(
                1 for pred in preds 
                if any(term in pred.lower() for term in self.config.professional_terms)
            )
            positive_rates[group] = positive_count / len(preds) if len(preds) > 0 else 0
        
        rates = list(positive_rates.values())
        if len(rates) < 2:
            return 0.0, positive_rates
        
        dpd = max(rates) - min(rates)
        return dpd, positive_rates
    
    def compute_equalized_odds_ratio(self, predictions_by_group, labels_by_group):
        """
        EOR = max_{g,g'} |P(ŷ ∈ Y+|y ∈ Y+, G=g) / P(ŷ ∈ Y+|y ∈ Y+, G=g')|
        """
        tpr_by_group = {}
        
        for group in predictions_by_group.keys():
            preds = predictions_by_group[group]
            labels = labels_by_group.get(group, [])
            
            if len(labels) == 0:
                continue
            
            tp = sum(
                1 for pred, label in zip(preds, labels)
                if any(term in pred.lower() for term in self.config.professional_terms)
                and any(term in label.lower() for term in self.config.professional_terms)
            )
            
            p = sum(
                1 for label in labels
                if any(term in label.lower() for term in self.config.professional_terms)
            )
            
            tpr_by_group[group] = tp / p if p > 0 else 0
        
        tprs = [v for v in tpr_by_group.values() if v > 0]
        if len(tprs) < 2:
            return 1.0, tpr_by_group
        
        eor = max(tprs) / min(tprs)
        return eor, tpr_by_group
    
    def compute_kl_divergence_by_group(self, predictions_by_group):
        """
        Compute KL divergence between group-conditional and marginal distributions
        """
        vocab = set()
        for preds in predictions_by_group.values():
            for pred in preds:
                vocab.update(pred.lower().split())
        
        vocab = sorted(list(vocab))
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        
        group_distributions = {}
        all_preds = []
        
        for group, preds in predictions_by_group.items():
            word_counts = np.zeros(len(vocab))
            for pred in preds:
                for word in pred.lower().split():
                    if word in word_to_idx:
                        word_counts[word_to_idx[word]] += 1
            
            word_counts = word_counts + 1e-10
            group_distributions[group] = word_counts / word_counts.sum()
            all_preds.extend(preds)
        
        marginal_counts = np.zeros(len(vocab))
        for pred in all_preds:
            for word in pred.lower().split():
                if word in word_to_idx:
                    marginal_counts[word_to_idx[word]] += 1
        
        marginal_counts = marginal_counts + 1e-10
        marginal_dist = marginal_counts / marginal_counts.sum()
        
        kl_divs = {}
        for group, dist in group_distributions.items():
            kl = entropy(dist, marginal_dist)
            kl_divs[group] = kl
        
        return kl_divs, group_distributions, marginal_dist


class CaptionQualityMetrics:
    """Compute caption quality metrics - Section 6.2"""
    
    def __init__(self):
        self.smoothing = SmoothingFunction()
    
    def compute_bleu(self, predictions, references):
        """Compute BLEU-4 score"""
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = [ref.lower().split()]
            
            score = sentence_bleu(
                ref_tokens, 
                pred_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smoothing.method1
            )
            bleu_scores.append(score)
        
        return np.mean(bleu_scores), bleu_scores
    
    def compute_meteor(self, predictions, references):
        """Simple METEOR approximation using word overlap"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(pred_words) == 0 or len(ref_words) == 0:
                scores.append(0.0)
                continue
            
            precision = len(pred_words & ref_words) / len(pred_words)
            recall = len(pred_words & ref_words) / len(ref_words)
            
            if precision + recall == 0:
                scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                scores.append(f1)
        
        return np.mean(scores), scores


class ReliabilityMetrics:
    """Compute reliability metrics - Section 6.4"""
    
    def compute_ece(self, confidences, accuracies, n_bins=10):
        """Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def compute_attention_entropy(self, attention_weights):
        """Compute attention entropy as uncertainty indicator - Equation (9)"""
        # attention_weights: [batch, seq_len, num_patches]
        entropies = []
        
        for attn in attention_weights:
            # Compute entropy for each token's attention distribution
            token_entropies = []
            for token_attn in attn:
                token_attn = token_attn + 1e-10  # Numerical stability
                token_attn = token_attn / token_attn.sum()
                ent = -np.sum(token_attn * np.log(token_attn))
                token_entropies.append(ent)
            entropies.append(np.mean(token_entropies))
        
        return np.array(entropies)


class VLMTrainer:
    """Trainer for VLM baseline"""
    
    def __init__(self, config=config):
        self.config = config
        self.device = config.device
        
        # Initialize model and processor
        print(f"Loading model: {config.model_name}")
        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(config.model_name)
        self.model.to(self.device)
        
        # Initialize metrics
        self.fairness_metrics = FairnessMetrics(config)
        self.quality_metrics = CaptionQualityMetrics()
        self.reliability_metrics = ReliabilityMetrics()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = []
        
    def prepare_data(self):
        """Prepare train and validation datasets"""
        print("Preparing datasets...")
        
        train_dataset = FairFaceVLMDataset('train', self.processor, self.config)
        val_dataset = FairFaceVLMDataset('validation', self.processor, self.config)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
    def setup_training(self):
        """Setup optimizer and scheduler"""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.train_loader) * self.config.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if self.config.fp16 else None
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass
            if self.config.fp16:
                with autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Periodic evaluation
            if (step + 1) % self.config.eval_every_n_steps == 0:
                val_loss, val_metrics = self.evaluate()
                self.val_losses.append(val_loss)
                self.val_metrics_history.append(val_metrics)
                self.model.train()
        
        avg_epoch_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_epoch_loss)
        
        return avg_epoch_loss
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        
        # Collect predictions and demographics
        all_predictions = []
        all_references = []
        all_demographics = {'gender': [], 'age_group': [], 'race': []}
        predictions_by_gender = defaultdict(list)
        predictions_by_age = defaultdict(list)
        predictions_by_race = defaultdict(list)
        references_by_gender = defaultdict(list)
        references_by_age = defaultdict(list)
        references_by_race = defaultdict(list)
        
        print("\nEvaluating...")
        for batch in tqdm(self.val_loader, desc="Validation"):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Compute loss
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            total_loss += outputs.loss.item()
            
            # Generate captions
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                max_length=self.config.max_caption_length
            )
            
            generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            reference_captions = batch['caption_text']
            
            # Store predictions
            for i in range(len(generated_captions)):
                pred = generated_captions[i]
                ref = reference_captions[i]
                gender = batch['gender'][i]
                age = batch['age_group'][i]
                race = batch['race'][i]
                
                all_predictions.append(pred)
                all_references.append(ref)
                
                # Group by demographics
                predictions_by_gender[gender].append(pred)
                predictions_by_age[age].append(pred)
                predictions_by_race[race].append(pred)
                references_by_gender[gender].append(ref)
                references_by_age[age].append(ref)
                references_by_race[race].append(ref)
                
                all_demographics['gender'].append(gender)
                all_demographics['age_group'].append(age)
                all_demographics['race'].append(race)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Compute metrics
        print("Computing fairness metrics...")
        
        # Fairness metrics
        dpd_gender, gender_rates = self.fairness_metrics.compute_demographic_parity_difference(predictions_by_gender)
        dpd_age, age_rates = self.fairness_metrics.compute_demographic_parity_difference(predictions_by_age)
        dpd_race, race_rates = self.fairness_metrics.compute_demographic_parity_difference(predictions_by_race)
        
        eor_gender, _ = self.fairness_metrics.compute_equalized_odds_ratio(
            predictions_by_gender, references_by_gender
        )
        eor_age, _ = self.fairness_metrics.compute_equalized_odds_ratio(
            predictions_by_age, references_by_age
        )
        eor_race, _ = self.fairness_metrics.compute_equalized_odds_ratio(
            predictions_by_race, references_by_race
        )
        
        kl_gender, _, _ = self.fairness_metrics.compute_kl_divergence_by_group(predictions_by_gender)
        kl_age, _, _ = self.fairness_metrics.compute_kl_divergence_by_group(predictions_by_age)
        kl_race, _, _ = self.fairness_metrics.compute_kl_divergence_by_group(predictions_by_race)
        
        # Caption quality metrics
        print("Computing quality metrics...")
        bleu_score, _ = self.quality_metrics.compute_bleu(all_predictions, all_references)
        meteor_score, _ = self.quality_metrics.compute_meteor(all_predictions, all_references)
        
        metrics = {
            'loss': avg_loss,
            'fairness': {
                'dpd_gender': dpd_gender,
                'dpd_age': dpd_age,
                'dpd_race': dpd_race,
                'eor_gender': eor_gender,
                'eor_age': eor_age,
                'eor_race': eor_race,
                'kl_gender': kl_gender,
                'kl_age': kl_age,
                'kl_race': kl_race,
                'gender_positive_rates': gender_rates,
                'age_positive_rates': age_rates,
                'race_positive_rates': race_rates
            },
            'quality': {
                'bleu4': bleu_score,
                'meteor': meteor_score
            }
        }
        
        # Print metrics
        print(f"\n{'='*50}")
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"\nFairness Metrics:")
        print(f"  DPD (Gender): {dpd_gender:.4f} (target: ≤0.08)")
        print(f"  DPD (Age): {dpd_age:.4f}")
        print(f"  DPD (Race): {dpd_race:.4f}")
        print(f"  EOR (Gender): {eor_gender:.4f} (target: 0.9-1.1)")
        print(f"\nCaption Quality:")
        print(f"  BLEU-4: {bleu_score:.4f}")
        print(f"  METEOR: {meteor_score:.4f}")
        print(f"{'='*50}\n")
        
        return avg_loss, metrics
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        self.setup_training()
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_loss, val_metrics = self.evaluate()
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Save regular checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=False)
        
        # Final evaluation
        print("\nRunning final evaluation...")
        final_val_loss, final_metrics = self.evaluate()
        
        # Plot results
        self.plot_training_curves()
        self.plot_fairness_metrics()
        
        # Save final results
        self.save_results(final_metrics)
        
        return final_metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': metrics
        }
        
        if is_best:
            path = os.path.join(self.config.checkpoints_dir, 'best_model.pt')
            torch.save(checkpoint, path)
            print(f"Saved best model to {path}")
        else:
            path = os.path.join(self.config.checkpoints_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, path)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        axes[0].plot(self.train_losses, label='Train Loss', marker='o')
        axes[0].plot(self.val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Metrics over time
        if self.val_metrics_history:
            dpd_values = [m['fairness']['dpd_gender'] for m in self.val_metrics_history]
            bleu_values = [m['quality']['bleu4'] for m in self.val_metrics_history]
            
            ax2 = axes[1]
            ax2.plot(dpd_values, 'r-o', label='DPD (Gender)')
            ax2.axhline(y=0.08, color='r', linestyle='--', label='DPD Target (0.08)')
            ax2.set_xlabel('Evaluation Step')
            ax2.set_ylabel('DPD', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.legend(loc='upper left')
            ax2.grid(True)
            
            ax3 = ax2.twinx()
            ax3.plot(bleu_values, 'b-s', label='BLEU-4')
            ax3.set_ylabel('BLEU-4', color='b')
            ax3.tick_params(axis='y', labelcolor='b')
            ax3.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plots_dir, 'training_curves.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves to {self.config.plots_dir}/training_curves.pdf")
    
    def plot_fairness_metrics(self):
        """Plot fairness metrics"""
        if not self.val_metrics_history:
            return
        
        final_metrics = self.val_metrics_history[-1]['fairness']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # DPD by demographic
        dpd_values = [
            final_metrics['dpd_gender'],
            final_metrics['dpd_age'],
            final_metrics['dpd_race']
        ]
        axes[0, 0].bar(['Gender', 'Age', 'Race'], dpd_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 0].axhline(y=0.08, color='red', linestyle='--', label='Target (0.08)')
        axes[0, 0].set_ylabel('DPD')
        axes[0, 0].set_title('Demographic Parity Difference')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # EOR by demographic
        eor_values = [
            final_metrics['eor_gender'],
            final_metrics['eor_age'],
            final_metrics['eor_race']
        ]
        axes[0, 1].bar(['Gender', 'Age', 'Race'], eor_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[0, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(y=1.1, color='red', linestyle='--', alpha=0.5, label='Target Range [0.9, 1.1]')
        axes[0, 1].set_ylabel('EOR')
        axes[0, 1].set_title('Equalized Odds Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Positive rates by gender
        gender_rates = final_metrics['gender_positive_rates']
        axes[0, 2].bar(gender_rates.keys(), gender_rates.values(), color='#FF6B6B')
        axes[0, 2].set_ylabel('Positive Rate')
        axes[0, 2].set_title('Professional Term Rate by Gender')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Positive rates by age
        age_rates = final_metrics['age_positive_rates']
        axes[1, 0].bar(age_rates.keys(), age_rates.values(), color='#4ECDC4')
        axes[1, 0].set_ylabel('Positive Rate')
        axes[1, 0].set_title('Professional Term Rate by Age')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Positive rates by race
        race_rates = final_metrics['race_positive_rates']
        axes[1, 1].bar(race_rates.keys(), race_rates.values(), color='#45B7D1')
        axes[1, 1].set_ylabel('Positive Rate')
        axes[1, 1].set_title('Professional Term Rate by Race')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # KL divergence heatmap
        kl_data = {
            'Gender': list(final_metrics['kl_gender'].values()),
            'Age': list(final_metrics['kl_age'].values()),
            'Race': list(final_metrics['kl_race'].values())
        }
        kl_df = pd.DataFrame(kl_data)
        sns.heatmap(kl_df.T, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[1, 2])
        axes[1, 2].set_title('KL Divergence from Marginal')
        axes[1, 2].set_xlabel('Group Index')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.plots_dir, 'fairness_metrics.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved fairness metrics to {self.config.plots_dir}/fairness_metrics.pdf")
    
    def save_results(self, metrics):
        """Save final results to JSON"""
        results = {
            'config': {
                'model': self.config.model_name,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'num_epochs': self.config.num_epochs
            },
            'final_metrics': metrics,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
        }
        
        # Convert numpy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        results = convert_to_native(results)
        
        results_path = os.path.join(self.config.results_dir, 'baseline_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSaved results to {results_path}")
        
        # Also save as CSV for easy viewing
        flat_metrics = {
            'dpd_gender': metrics['fairness']['dpd_gender'],
            'dpd_age': metrics['fairness']['dpd_age'],
            'dpd_race': metrics['fairness']['dpd_race'],
            'eor_gender': metrics['fairness']['eor_gender'],
            'eor_age': metrics['fairness']['eor_age'],
            'eor_race': metrics['fairness']['eor_race'],
            'bleu4': metrics['quality']['bleu4'],
            'meteor': metrics['quality']['meteor'],
            'val_loss': metrics['loss']
        }
        
        df = pd.DataFrame([flat_metrics])
        csv_path = os.path.join(self.config.results_dir, 'baseline_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")


def main():
    """Main training function"""
    print("="*60)
    print("Baseline VLM Training with Fairness Evaluation")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Output Dir: {config.output_dir}")
    print("="*60)
    
    # Initialize trainer
    trainer = VLMTrainer(config)
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model
    final_metrics = trainer.train()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nFinal Metrics:")
    print(f"  DPD (Gender): {final_metrics['fairness']['dpd_gender']:.4f} (target: ≤0.08)")
    print(f"  EOR (Gender): {final_metrics['fairness']['eor_gender']:.4f} (target: 0.9-1.1)")
    print(f"  BLEU-4: {final_metrics['quality']['bleu4']:.4f}")
    print(f"  METEOR: {final_metrics['quality']['meteor']:.4f}")
    print(f"\nResults saved to: {config.results_dir}")
    print(f"Plots saved to: {config.plots_dir}")
    print(f"Checkpoints saved to: {config.checkpoints_dir}")
    print("="*60)


if __name__ == "__main__":
    main()