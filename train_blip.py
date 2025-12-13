# pip install --user pycocotools matplotlib seaborn pandas numpy pillow wordcloud textblob scikit-learn
# pip install --user accelerate datasets evaluate
# pip install --user transformers torch torchvision pillow nltk rouge-score
# pip install --user git+https://github.com/tylin/coco-caption.git

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from PIL import Image
import os
from tqdm import tqdm
import json
from collections import defaultdict
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# Cell: Download FairFace from Hugging Face (CORRECTED)

from datasets import load_dataset
import os
import pandas as pd

FAIRFACE_BASE_DIR = "/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/fairface"
os.makedirs(FAIRFACE_BASE_DIR, exist_ok=True)

print("Downloading FairFace from Hugging Face...")
print("Using '0.25' config (face padding margin of 0.25)")

# Load the dataset with config
dataset = load_dataset("HuggingFaceM4/FairFace", "0.25", cache_dir=FAIRFACE_BASE_DIR)

print(f"\nTrain samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")

# Explore the structure
print("\nDataset structure:")
print(dataset)

print("\nExample from training set:")
example = dataset['train'][0]
print("Keys:", example.keys())
print("Age:", example.get('age'))
print("Gender:", example.get('gender'))
print("Race:", example.get('race'))
print("Image shape:", example['image'].size if hasattr(example['image'], 'size') else type(example['image']))

# Convert to pandas for easier handling
print("\nConverting to pandas DataFrames...")
train_df = pd.DataFrame(dataset['train'])
val_df = pd.DataFrame(dataset['validation'])

print("\nTraining DataFrame:")
print(train_df.head())
print("\nColumns:", train_df.columns.tolist())

print("\nGender distribution (train):")
print(train_df['gender'].value_counts())

print("\nRace distribution (train):")
print(train_df['race'].value_counts())

print("\nAge distribution (train):")
print(train_df['age'].value_counts())

# Save to CSV
print("\nSaving metadata to CSV...")
train_df[['age', 'gender', 'race']].to_csv(
    os.path.join(FAIRFACE_BASE_DIR, 'fairface_train_labels.csv'), 
    index=True
)
val_df[['age', 'gender', 'race']].to_csv(
    os.path.join(FAIRFACE_BASE_DIR, 'fairface_val_labels.csv'), 
    index=True
)

print("\n✓ FairFace dataset loaded successfully!")
print(f"Data cached at: {FAIRFACE_BASE_DIR}")


# Cell: Load Cached FairFace Dataset

from datasets import load_dataset

FAIRFACE_BASE_DIR = "/nfs/roberts/project/cpsc4710/cpsc4710_ag2995/project/datasets/fairface"

print("Loading cached FairFace dataset...")
dataset = load_dataset("HuggingFaceM4/FairFace", "0.25", cache_dir=FAIRFACE_BASE_DIR)

train_hf = dataset['train']
val_hf = dataset['validation']

print(f"Training samples: {len(train_hf):,}")
print(f"Validation samples: {len(val_hf):,}")

# Show example
print("\nExample sample:")
example = train_hf[0]
print(f"  Keys: {example.keys()}")
print(f"  Gender: {example['gender']}")
print(f"  Race: {example['race']}")
print(f"  Age: {example['age']}")
print(f"  Image type: {type(example['image'])}")


# Cell: Create FairFace Dataset Class
from collections import Counter, defaultdict
class FairFaceHFDataset(Dataset):
    """FairFace Dataset from Hugging Face with caption generation"""
    
    def __init__(self, hf_dataset, processor, max_length=50, mode='train'):
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
        """Generate captions using ALL 3 available FairFace attributes"""
        
        age_map = {
            '0-2': 'infant', '3-9': 'child', '10-19': 'teenager',
            '20-29': 'young adult', '30-39': 'adult',
            '40-49': 'middle-aged adult', '50-59': 'middle-aged adult',
            '60-69': 'senior', 'more than 70': 'elderly person'
        }
        
        age_desc = age_map.get(age, 'adult')
        gender_noun = 'man' if gender == 'Male' else 'woman'
        
        # 5 captions with different attribute combinations
        captions = [
            # Full: Age + Race + Gender (ALL 3 attributes)
            f"a photo of a {age_desc} {race} {gender_noun}",
            
            # # Age + Gender only
            # f"a portrait of a {age_desc} {gender_noun}",
            
            # # Race + Gender only  
            # f"a close-up photo of a {race} {gender_noun}",
            
            # # Gender only
            # f"a headshot of a {gender_noun}",
            
            # # Neutral (no demographics)
            # f"a photo of a person"
        ]
        
        return captions
    
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
        
        # Generate captions
        captions = self.generate_caption(age, gender, race)
        
        # Pick one random caption for training
        caption = np.random.choice(captions)
        
        # Process image
        encoding = self.processor(
            images=image,
            return_tensors="pt"
        )
        
        # Process text
        text_encoding = self.processor.tokenizer(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'gender': gender,
            'race': race,
            'age': age,
            'captions': captions  # Store all captions for eval
        }

print("Creating datasets...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

train_dataset = FairFaceHFDataset(train_hf, processor, mode='train')
val_dataset = FairFaceHFDataset(val_hf, processor, mode='val')


# Cell: Setup Training Configuration

# Load model
print("Loading BLIP model...")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Training hyperparameters
BATCH_SIZE = 32  # FairFace images are small, can use larger batch
EPOCHS = 5
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

print(f"\nTraining Configuration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Total steps: {total_steps:,}")
print(f"  Steps per epoch: {len(train_loader):,}")
print(f"  Warmup steps: {WARMUP_STEPS}")

# Cell: Training Functions

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # Move to device
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Create labels
        labels = input_ids.clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(step+1):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
    
    return total_loss / len(dataloader)
    

# Cell: Training Loop

print("\nStarting training on FairFace dataset...")
print("=" * 80)

best_val_loss = float('inf')
training_history = {
    'train_loss': [],
    'val_loss': []
}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 80)
    
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
    
    # Validate
    val_loss = validate(model, val_loader, device)
    
    # Save history
    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Training Loss:   {train_loss:.4f}")
    print(f"  Validation Loss: {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"  ✓ New best model! Saving...")
        model.save_pretrained("checkpoints/blip-finetuned-fairface")
        processor.save_pretrained("checkpoints/blip-finetuned-fairface")
    
    print("-" * 80)

print("\n" + "=" * 80)
print("Training completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("=" * 80)

# Save training history
with open('checkpoints/training_history.json', 'w') as f:
    json.dump(training_history, f)

