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