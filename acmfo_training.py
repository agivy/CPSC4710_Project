"""
Adaptive Cross-Modal Fairness Optimizer (ACMFO) - COCO Training Implementation
FIXED VERSION - Device mismatch and shape errors corrected

Based on proposal: "Fairness-Aware Cross-Modal Interpretability for Vision-Language Models"

Implements:
1. Multi-objective loss: L_ACMFO = L_task + λ₁·L_fairness + λ₂·L_cross-modal
2. Demographic parity via KL divergence regularization (using CLIP-inferred demographics)
3. Cross-modal disentanglement via mutual information minimization
4. Training on MS-COCO with CLIP demographic inference
5. Evaluation on MS-COCO validation set

Target: DPD ≤ 0.08, BLEU-4/ROUGE-L within 5% of baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
from transformers import get_linear_schedule_with_warmup
from PIL import Image
import os
from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict, Counter
from pycocotools.coco import COCO
from sklearn.cluster import KMeans

# Evaluation metrics
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ============================================================================
# Configuration
# ============================================================================

class ACMFOConfig:
    """ACMFO Training Configuration"""
    
    # Paths
    COCO_BASE_DIR = "/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/coco2017"
    OUTPUT_DIR = "acmfo_coco_checkpoints"
    
    # Model
    BASE_MODEL = "Salesforce/blip-image-captioning-base"
    
    # Training hyperparameters
    BATCH_SIZE = 8  # Reduced due to CLIP inference overhead
    EPOCHS = 3
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 500
    MAX_GRAD_NORM = 1.0
    
    # Data sampling (use subset for faster training)
    TRAIN_SAMPLES = 20000  # Use 20k for training (full is ~118k)
    VAL_SAMPLES = 5000
    
    # ACMFO hyperparameters (from proposal)
    LAMBDA_FAIRNESS = 1.0      # λ₁: fairness loss weight
    LAMBDA_CROSS_MODAL = 0.05   # λ₂: cross-modal disentanglement weight
    BETA_MI = 1.0              # β: MI preservation weight
    
    # Fairness parameters
    NUM_CAPTION_CLUSTERS = 100  # k-means clusters for KL approximation
    EMA_DECAY = 0.99           # Exponential moving average for group distributions
    GENDER_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for gender inference
    
    # Target metrics
    TARGET_DPD = 0.08
    TARGET_PERFORMANCE_RETENTION = 0.95

config = ACMFOConfig()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


# ============================================================================
# COCO Dataset with CLIP Gender Inference
# ============================================================================

class COCOWithDemographics(Dataset):
    """MS-COCO dataset with CLIP-based demographic inference"""
    
    def __init__(self, coco_root, annotation_file, blip_processor, mode='train', max_samples=None):
        self.coco_root = coco_root
        self.coco = COCO(annotation_file)
        self.blip_processor = blip_processor
        self.mode = mode
        
        # Get image IDs
        all_image_ids = list(self.coco.imgs.keys())
        
        # Sample if needed
        if max_samples and max_samples < len(all_image_ids):
            np.random.seed(42)
            self.image_ids = np.random.choice(all_image_ids, max_samples, replace=False).tolist()
        else:
            self.image_ids = all_image_ids
        
        print(f"\n{mode.upper()} Dataset:")
        print(f"  Total images: {len(self.image_ids):,}")
        
        # Load CLIP for demographic inference
        print(f"  Loading CLIP for demographic inference...")
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True
        ).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        print(f"  ✓ CLIP loaded")
    
    def infer_gender(self, image):
        """Infer gender using CLIP zero-shot classification"""
        try:
            texts = ["a man", "a woman", "a male person", "a female person", "men", "women"]
            
            inputs = self.clip_processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)[0]
            
            # Aggregate male/female probabilities
            male_prob = (probs[0] + probs[2] + probs[4]) / 3
            female_prob = (probs[1] + probs[3] + probs[5]) / 3
            
            # Normalize
            total = male_prob + female_prob
            male_prob = male_prob / total
            female_prob = female_prob / total
            
            if male_prob > female_prob:
                return {"gender": "Male", "confidence": male_prob.item()}
            else:
                return {"gender": "Female", "confidence": female_prob.item()}
        except Exception as e:
            return {"gender": "unknown", "confidence": 0.0}
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.coco_root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get captions
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]
        
        # Use first caption for training
        caption = captions[0]
        
        # Infer demographics
        demographics = self.infer_gender(image)
        
        # Process image for BLIP
        encoding = self.blip_processor(
            images=image,
            return_tensors="pt"
        )
        
        # Process text for BLIP
        text_encoding = self.blip_processor.tokenizer(
            caption,
            padding="max_length",
            max_length=50,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'gender': demographics['gender'],
            'confidence': demographics['confidence'],
            'all_captions': captions  # For validation
        }


def collate_fn(batch):
    """Custom collate function"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    genders = [item['gender'] for item in batch]
    confidences = [item['confidence'] for item in batch]
    
    # For validation
    all_captions = [item['all_captions'] for item in batch] if 'all_captions' in batch[0] else None
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'genders': genders,
        'confidences': confidences,
        'all_captions': all_captions
    }


# ============================================================================
# ACMFO Loss Components
# ============================================================================

class FairnessLoss(nn.Module):
    """
    Demographic Parity Regularization via KL Divergence
    
    L_fairness = Σ_g w_g · KL(P(Y|X,G=g) || P(Y|X))
    """
    
    def __init__(self, num_clusters=100, ema_decay=0.99):
        super().__init__()
        self.num_clusters = num_clusters
        self.ema_decay = ema_decay
        
        self.register_buffer('cluster_centers', None)
        self.register_buffer('ema_marginal', None)
        self.register_buffer('ema_male', None)
        self.register_buffer('ema_female', None)
        
        self.initialized = False
    
    def initialize_clusters(self, caption_embeddings):
        """Initialize k-means clusters on caption embeddings"""
        print("Initializing caption clusters for fairness loss...")
        
        # Get device from input
        target_device = caption_embeddings.device
        
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        kmeans.fit(caption_embeddings.cpu().numpy())
        
        # FIXED: Ensure proper dtype and device for cluster centers
        self.cluster_centers = torch.from_numpy(kmeans.cluster_centers_).float().to(target_device)
        
        # FIXED: Initialize EMA buffers on correct device
        uniform_dist = torch.ones(self.num_clusters, device=target_device, dtype=torch.float32) / self.num_clusters
        self.ema_marginal = uniform_dist.clone()
        self.ema_male = uniform_dist.clone()
        self.ema_female = uniform_dist.clone()
        
        self.initialized = True
        print(f"✓ Initialized {self.num_clusters} caption clusters")
    
    def assign_to_clusters(self, embeddings):
        """Assign embeddings to nearest cluster"""
        # FIXED: Ensure embeddings are float32 to match cluster_centers
        embeddings = embeddings.float()
        distances = torch.cdist(embeddings, self.cluster_centers)
        assignments = distances.argmin(dim=1)
        return assignments
    
    def compute_distribution(self, assignments):
        """Compute empirical distribution over clusters"""
        counts = torch.bincount(assignments, minlength=self.num_clusters).float()
        dist = counts / (counts.sum() + 1e-10)
        return dist
    
    def forward(self, caption_embeddings, genders, confidences, group_weights, threshold=0.3):
        """
        Args:
            caption_embeddings: [batch_size, embedding_dim]
            genders: list of 'Male'/'Female'/'unknown'
            confidences: list of confidence scores
            group_weights: dict with inverse frequency weights
            threshold: minimum confidence to include sample
        """
        if not self.initialized:
            return torch.tensor(0.0, device=caption_embeddings.device)
        
        # FIXED: Ensure proper device for mask tensor
        valid_mask = torch.tensor(
            [(g in ['Male', 'Female']) and (c > threshold) 
             for g, c in zip(genders, confidences)],
            device=caption_embeddings.device,
            dtype=torch.bool
        )
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=caption_embeddings.device)
        
        # Get valid samples
        valid_embeddings = caption_embeddings[valid_mask]
        valid_genders = [g for i, g in enumerate(genders) if valid_mask[i]]
        
        # Assign to clusters
        assignments = self.assign_to_clusters(valid_embeddings)
        
        # Compute marginal distribution
        marginal_dist = self.compute_distribution(assignments)
        
        # Update EMA
        with torch.no_grad():
            self.ema_marginal = self.ema_decay * self.ema_marginal + (1 - self.ema_decay) * marginal_dist
        
        # Compute group-conditional distributions
        male_mask = torch.tensor([g == 'Male' for g in valid_genders], device=assignments.device, dtype=torch.bool)
        female_mask = torch.tensor([g == 'Female' for g in valid_genders], device=assignments.device, dtype=torch.bool)
        
        kl_losses = []
        
        if male_mask.sum() > 0:
            male_dist = self.compute_distribution(assignments[male_mask])
            with torch.no_grad():
                self.ema_male = self.ema_decay * self.ema_male + (1 - self.ema_decay) * male_dist
            
            # FIXED: Swap arguments in kl_div (input should be log-probabilities)
            kl_male = F.kl_div(
                torch.log(self.ema_male + 1e-10),
                self.ema_marginal,
                reduction='batchmean',
                log_target=False
            )
            kl_losses.append(group_weights.get('Male', 1.0) * kl_male)
        
        if female_mask.sum() > 0:
            female_dist = self.compute_distribution(assignments[female_mask])
            with torch.no_grad():
                self.ema_female = self.ema_decay * self.ema_female + (1 - self.ema_decay) * female_dist
            
            # FIXED: Swap arguments in kl_div (input should be log-probabilities)
            kl_female = F.kl_div(
                torch.log(self.ema_female + 1e-10),
                self.ema_marginal,
                reduction='batchmean',
                log_target=False
            )
            kl_losses.append(group_weights.get('Female', 1.0) * kl_female)
        
        if len(kl_losses) == 0:
            return torch.tensor(0.0, device=caption_embeddings.device)
        
        fairness_loss = sum(kl_losses) / len(kl_losses)
        return fairness_loss


class CrossModalDisentanglementLoss(nn.Module):
    """
    Mutual Information Minimization for Cross-Modal Disentanglement
    """
    
    def __init__(self, embedding_dim=768, hidden_dim=256, beta=1.0):
        super().__init__()
        self.beta = beta
        self.hidden_dim = hidden_dim
        
        # Don't create fixed estimators - we'll create them dynamically
        # based on actual feature dimensions
        self.mi_estimator_bias = None
        self.mi_estimator_task = None
    
    def get_or_create_estimator(self, input_dim, estimator_type='bias'):
        """Create MI estimator with correct input dimension"""
        if estimator_type == 'bias':
            if self.mi_estimator_bias is None or self.mi_estimator_bias[0].in_features != input_dim:
                self.mi_estimator_bias = nn.Sequential(
                    nn.Linear(input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1)
                ).to(next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cuda')
            return self.mi_estimator_bias
        else:  # task
            if self.mi_estimator_task is None or self.mi_estimator_task[0].in_features != input_dim:
                self.mi_estimator_task = nn.Sequential(
                    nn.Linear(input_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1)
                ).to(next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cuda')
            return self.mi_estimator_task
    
    def compute_mi_infonce(self, v, t, estimator):
        """Estimate MI using InfoNCE lower bound"""
        batch_size = v.shape[0]
        device = v.device
        
        # Positive pairs
        pos_pairs = torch.cat([v, t], dim=1)  # [B, 2*D]
        pos_scores = estimator(pos_pairs)  # [B, 1]
        
        # Negative sampling: pair each v with all t's
        v_expanded = v.unsqueeze(1).expand(batch_size, batch_size, -1)  # [B, B, D]
        t_expanded = t.unsqueeze(0).expand(batch_size, batch_size, -1)  # [B, B, D]
        neg_pairs = torch.cat([v_expanded, t_expanded], dim=2)  # [B, B, 2D]
        neg_pairs = neg_pairs.reshape(batch_size * batch_size, -1)  # [B*B, 2D]
        neg_scores = estimator(neg_pairs).view(batch_size, batch_size)  # [B, B]
        
        # InfoNCE: E[log(exp(pos) / sum(exp(neg)))]
        mi_estimate = pos_scores.mean() - torch.logsumexp(neg_scores, dim=1).mean()
        
        return mi_estimate
    
    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: [batch_size, num_patches, dim]
            text_features: [batch_size, seq_len, dim]
        """
        device = visual_features.device
        
        v_pooled = visual_features.mean(dim=1).float()  # [batch_size, dim]
        t_pooled = text_features.mean(dim=1).float()    # [batch_size, dim]
        
        # Determine split point based on actual embedding dimension
        embedding_dim = v_pooled.shape[1]
        split_point = min(50, embedding_dim // 2)  # Use at most 50 dims or half, whichever is smaller
        
        v_bias = v_pooled[:, :split_point]
        v_task = v_pooled[:, split_point:]
        t_bias = t_pooled[:, :split_point]
        t_task = t_pooled[:, split_point:]
        
        # Get or create estimators with correct dimensions
        bias_input_dim = v_bias.shape[1] * 2  # doubled because we concat v and t
        task_input_dim = v_task.shape[1] * 2
        
        estimator_bias = self.get_or_create_estimator(bias_input_dim, 'bias')
        estimator_task = self.get_or_create_estimator(task_input_dim, 'task')
        
        # MI estimates
        mi_bias = self.compute_mi_infonce(v_bias, t_bias, estimator_bias)
        mi_task = self.compute_mi_infonce(v_task, t_task, estimator_task)
        
        # We want to maximize MI(bias) and minimize MI(task)
        # In a minimization framework: minimize -MI(bias) + beta*MI(task)
        loss = -mi_bias + self.beta * mi_task
        
        return loss


# ============================================================================
# ACMFO Trainer
# ============================================================================

class ACMFOTrainer:
    """ACMFO training on MS-COCO"""
    
    def __init__(self, config):
        self.config = config
        
        print("\nLoading BLIP model...")
        self.processor = BlipProcessor.from_pretrained(config.BASE_MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(
            config.BASE_MODEL,
            use_safetensors=True
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        
        # Loss components
        self.fairness_loss_fn = FairnessLoss(
            num_clusters=config.NUM_CAPTION_CLUSTERS,
            ema_decay=config.EMA_DECAY
        ).to(device)
        
        self.cross_modal_loss_fn = CrossModalDisentanglementLoss(
            embedding_dim=768,
            beta=config.BETA_MI
        ).to(device)
        
        # Optimizer - note: cross_modal_loss_fn parameters are added dynamically
        # We'll update optimizer in training loop as needed
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + 
            list(self.fairness_loss_fn.parameters()),
            lr=config.LEARNING_RATE
        )
        
        self.cross_modal_optimizers_initialized = False
        
        self.scheduler = None
        self.group_weights = {'Male': 1.0, 'Female': 1.0}
        
        # History
        self.history = {
            'train_loss': [],
            'train_task_loss': [],
            'train_fairness_loss': [],
            'train_cross_modal_loss': [],
            'val_bleu4': [],
            'val_rouge_l': [],
            'val_dpd': []
        }
    
    def compute_group_weights(self, dataloader):
        """Compute inverse frequency weights"""
        print("\nComputing group weights...")
        genders = []
        
        for batch in tqdm(dataloader, desc="Collecting demographics"):
            genders.extend([g for g, c in zip(batch['genders'], batch['confidences']) 
                          if g in ['Male', 'Female'] and c > self.config.GENDER_CONFIDENCE_THRESHOLD])
        
        counts = Counter(genders)
        total = len(genders)
        
        if total == 0:
            return {'Male': 1.0, 'Female': 1.0}
        
        weights = {gender: total / count for gender, count in counts.items()}
        max_weight = max(weights.values())
        weights = {k: v / max_weight for k, v in weights.items()}
        
        print(f"Group weights: {weights}")
        return weights
    
    def initialize_fairness_clusters(self, dataloader):
        """Initialize caption clusters"""
        print("\nInitializing fairness clusters...")
        
        self.model.eval()
        caption_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Collecting embeddings"):
                input_ids = batch['input_ids'].to(device)
                
                # FIXED: Get text embeddings properly from BLIP
                # BLIP uses a text decoder based on BERT
                if hasattr(self.model.text_decoder, 'bert'):
                    text_outputs = self.model.text_decoder.bert.embeddings(input_ids)
                else:
                    # Fallback: use text encoder if available
                    text_outputs = self.model.text_encoder(input_ids).last_hidden_state
                
                text_pooled = text_outputs.mean(dim=1)
                
                caption_embeddings.append(text_pooled.cpu())
        
        caption_embeddings = torch.cat(caption_embeddings, dim=0)
        print(f"Collected {len(caption_embeddings)} embeddings")
        
        # FIXED: Move back to device before initializing
        self.fairness_loss_fn.initialize_clusters(caption_embeddings.to(device))
        self.model.train()
    
    def train_epoch(self, dataloader, epoch):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        total_task = 0
        total_fair = 0
        total_cm = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            genders = batch['genders']
            confidences = batch['confidences']
            
            # Labels
            labels = input_ids.clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            # Forward
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            task_loss = outputs.loss
            
            # FIXED: Get vision features properly
            with torch.no_grad():
                vision_outputs = self.model.vision_model(pixel_values, return_dict=True)
                vision_hidden = vision_outputs.last_hidden_state
                
                # Get text embeddings for fairness loss
                if hasattr(self.model.text_decoder, 'bert'):
                    text_embeds = self.model.text_decoder.bert.embeddings(input_ids)
                else:
                    text_embeds = self.model.text_encoder(input_ids).last_hidden_state
                
                text_pooled = text_embeds.mean(dim=1)
            
            # Fairness loss
            fairness_loss = self.fairness_loss_fn(
                text_pooled,
                genders,
                confidences,
                self.group_weights,
                threshold=self.config.GENDER_CONFIDENCE_THRESHOLD
            )
            
            # FIXED: Get text hidden states with gradient for cross-modal loss
            # We need to run the decoder to get hidden states
            if hasattr(self.model.text_decoder, 'bert'):
                decoder_outputs = self.model.text_decoder.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                text_hidden = decoder_outputs.last_hidden_state
            else:
                text_hidden = self.model.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            
            cross_modal_loss = self.cross_modal_loss_fn(vision_hidden, text_hidden)
            
            # After first forward pass, add cross-modal estimator params to optimizer
            if not self.cross_modal_optimizers_initialized:
                cross_modal_params = []
                if self.cross_modal_loss_fn.mi_estimator_bias is not None:
                    cross_modal_params.extend(list(self.cross_modal_loss_fn.mi_estimator_bias.parameters()))
                if self.cross_modal_loss_fn.mi_estimator_task is not None:
                    cross_modal_params.extend(list(self.cross_modal_loss_fn.mi_estimator_task.parameters()))
                
                if len(cross_modal_params) > 0:
                    # Add new parameter group to existing optimizer
                    self.optimizer.add_param_group({
                        'params': cross_modal_params,
                        'lr': self.config.LEARNING_RATE
                    })
                    self.cross_modal_optimizers_initialized = True
            
            # Combined loss
            loss = (task_loss + 
                   self.config.LAMBDA_FAIRNESS * fairness_loss +
                   self.config.LAMBDA_CROSS_MODAL * cross_modal_loss)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Log
            total_loss += loss.item()
            total_task += task_loss.item()
            total_fair += fairness_loss.item()
            total_cm += cross_modal_loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'task': f'{task_loss.item():.4f}',
                'fair': f'{fairness_loss.item():.4f}',
                'cm': f'{cross_modal_loss.item():.4f}'
            })
        
        n = len(dataloader)
        return {
            'loss': total_loss / n,
            'task_loss': total_task / n,
            'fairness_loss': total_fair / n,
            'cross_modal_loss': total_cm / n
        }
    
    def validate(self, dataloader):
        """Validate with metrics"""
        self.model.eval()
        
        all_generated = []
        all_references = []
        all_genders = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                pixel_values = batch['pixel_values'].to(device)
                
                generated_ids = self.model.generate(
                    pixel_values=pixel_values,
                    max_length=50,
                    num_beams=3
                )
                
                generated = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                all_generated.extend(generated)
                all_references.extend(batch['all_captions'])
                all_genders.extend(batch['genders'])
        
        metrics = self.compute_metrics(all_generated, all_references, all_genders)
        self.model.train()
        return metrics
    
    def compute_metrics(self, generated, references, genders):
        """Compute BLEU, ROUGE, DPD"""
        # BLEU-4
        refs_tok = [[ref.lower().split() for ref in refs] for refs in references]
        hyps_tok = [gen.lower().split() for gen in generated]
        
        smoothing = SmoothingFunction()
        bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(0.25,0.25,0.25,0.25), 
                           smoothing_function=smoothing.method1)
        
        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = []
        for gen, refs in zip(generated, references):
            scores = [scorer.score(ref, gen)['rougeL'].fmeasure for ref in refs]
            rouge_scores.append(max(scores))
        rouge_l = np.mean(rouge_scores)
        
        # DPD
        professional_words = ['doctor', 'engineer', 'scientist', 'executive', 'ceo', 'lawyer']
        
        male_caps = [g for g, gender in zip(generated, genders) if gender == 'Male']
        female_caps = [g for g, gender in zip(generated, genders) if gender == 'Female']
        
        if len(male_caps) > 0 and len(female_caps) > 0:
            male_prof = sum(any(w in c.lower() for w in professional_words) for c in male_caps) / len(male_caps)
            female_prof = sum(any(w in c.lower() for w in professional_words) for c in female_caps) / len(female_caps)
            dpd = abs(male_prof - female_prof)
        else:
            dpd = 0.0
        
        return {'bleu4': bleu4, 'rouge_l': rouge_l, 'dpd': dpd}
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print("\n" + "="*80)
        print("ACMFO TRAINING ON MS-COCO")
        print("="*80)
        
        # Setup
        total_steps = len(train_loader) * self.config.EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Compute group weights
        self.group_weights = self.compute_group_weights(train_loader)
        
        # Initialize fairness
        self.initialize_fairness_clusters(train_loader)
        
        best_metric = 0.0
        
        for epoch in range(self.config.EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.EPOCHS}")
            print("-" * 80)
            
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            # Log
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_task_loss'].append(train_metrics['task_loss'])
            self.history['train_fairness_loss'].append(train_metrics['fairness_loss'])
            self.history['train_cross_modal_loss'].append(train_metrics['cross_modal_loss'])
            self.history['val_bleu4'].append(val_metrics['bleu4'])
            self.history['val_rouge_l'].append(val_metrics['rouge_l'])
            self.history['val_dpd'].append(val_metrics['dpd'])
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val BLEU-4: {val_metrics['bleu4']:.4f}")
            print(f"  Val ROUGE-L: {val_metrics['rouge_l']:.4f}")
            print(f"  Val DPD: {val_metrics['dpd']:.4f}")
            
            # Save best
            composite = val_metrics['bleu4'] - val_metrics['dpd']
            if composite > best_metric:
                best_metric = composite
                print(f"  ✓ Best model! Saving...")
                self.save_checkpoint(epoch, val_metrics)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
    
    def save_checkpoint(self, epoch, metrics):
        """Save checkpoint"""
        checkpoint_path = os.path.join(self.config.OUTPUT_DIR, 'best_model')
        self.model.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        
        with open(os.path.join(self.config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
        
        with open(os.path.join(self.config.OUTPUT_DIR, 'best_metrics.json'), 'w') as f:
            json.dump({'epoch': epoch, **metrics}, f, indent=2)


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ACMFO: MS-COCO TRAINING WITH FAIRNESS OPTIMIZATION")
    print("="*80)
    
    # Load datasets
    print("\nLoading MS-COCO datasets...")
    train_ann = os.path.join(config.COCO_BASE_DIR, 'annotations/captions_train2017.json')
    val_ann = os.path.join(config.COCO_BASE_DIR, 'annotations/captions_val2017.json')
    train_dir = os.path.join(config.COCO_BASE_DIR, 'train2017')
    val_dir = os.path.join(config.COCO_BASE_DIR, 'val2017')
    
    processor = BlipProcessor.from_pretrained(config.BASE_MODEL)
    
    train_dataset = COCOWithDemographics(
        train_dir, train_ann, processor, 
        mode='train', max_samples=config.TRAIN_SAMPLES
    )
    
    val_dataset = COCOWithDemographics(
        val_dir, val_ann, processor,
        mode='val', max_samples=config.VAL_SAMPLES
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # CLIP needs main process
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Train
    trainer = ACMFOTrainer(config)
    trainer.train(train_loader, val_loader)
    
    print(f"\n✓ Model saved to: {config.OUTPUT_DIR}/best_model")


if __name__ == "__main__":
    main()