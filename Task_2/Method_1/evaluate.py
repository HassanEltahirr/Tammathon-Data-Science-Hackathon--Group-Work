# ========== IMPORTS ==========
import os
import random # Import random
import numpy as np
import pandas as pd
from PIL import Image
import time
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm # Use tqdm.notebook for Kaggle environment
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm # For EfficientNet and other modern architectures
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast # Automatic Mixed Precision

# ========== CONFIGURATION ==========
# --- Reproducibility ---
SEED = 42 # Choose any integer for your seed

# --- Training Params ---
EPOCHS = 15 # Increased epochs for better convergence
LR = 3e-5   # Adjusted learning rate, often lower for fine-tuning larger models
BATCH_SIZE = 32 # Adjust based on GPU memory
IMG_SIZE = 300 # EfficientNet-B3 works well with this size
MODEL_NAME = 'efficientnet_b3' # Using EfficientNet-B3
WEIGHT_DECAY = 0.01
NUM_WORKERS = 2 # Or os.cpu_count() // 2

# --- Paths & Device ---
DATA_DIR = "/kaggle/input/tammathon-task-2"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
VAL_IMG_DIR = os.path.join(DATA_DIR, "val")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
MODEL_SAVE_PATH = "/kaggle/working/best_passport_model.pth"
SUBMISSION_PATH = "/kaggle/working/submission.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ========== SEED FIXING FUNCTION ==========
def seed_everything(seed=42):
    """Sets seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # if using CUDA
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    # The two lines below might slow down training but ensure reproducibility for certain CUDA ops
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {seed} for reproducibility.")

# Call the seed function early
seed_everything(SEED)

# ========== DataLoader Worker Seeding Function ==========
def seed_worker(worker_id):
    """Sets seed for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ========== DATASET ==========
class PassportDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_test=False):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_name = self.data.iloc[idx]['path']
            img_path = os.path.join(self.root_dir, img_name)
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name} at index {idx}: {e}")
            # Return a dummy image and label/placeholder if error occurs
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = 'grey')
            if self.is_test:
                 if self.transform: image = self.transform(image)
                 return image
            else:
                label = 0 # Or handle differently
                if self.transform: image = self.transform(image)
                return image, label


        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image
        else:
            label = self.data.iloc[idx]['label']
            return image, label

# ========== TRANSFORMS ==========
# Mean and Std deviation for ImageNet models
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), # Slight zoom/translation
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0), # Cutout/Random Erasing
])

# Validation and Test transforms (no augmentation, just resize and normalize)
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# TTA Transforms (Example: Original + Horizontal Flip)
tta_transforms = [
    val_test_transform, # Original
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0), # Force flip - Note: RandomFlip is still random, but seeding makes it predictable
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
]


# ========== SAMPLER (for class imbalance) ==========
train_df = pd.read_csv(TRAIN_CSV)
train_labels = train_df['label'].values
class_counts = np.bincount(train_labels)
if len(class_counts) < 2:
     print("Warning: Only one class found in training data. Sampler may not be effective.")
     sampler = None # Fallback to no sampler
else:
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[label] for label in train_labels])
    # Create a generator for the sampler to ensure its randomness is also seeded
    g = torch.Generator()
    g.manual_seed(SEED)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=g # Use the seeded generator
    )
    print(f"Class Counts: {class_counts}")
    print(f"Calculated Pos Weight (0/1): {class_counts[0] / class_counts[1]:.4f}")

# ========== DATALOADERS ==========
train_dataset = PassportDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=train_transform)
val_dataset = PassportDataset(VAL_CSV, VAL_IMG_DIR, transform=val_test_transform)

# Use sampler only for training loader and add worker_init_fn
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True, # Helps speed up CPU to GPU transfer
    worker_init_fn=seed_worker # Add worker seeding
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2, # Can often use larger batch size for validation
    shuffle=False, # No need to shuffle validation data
    num_workers=NUM_WORKERS,
    pin_memory=True,
    worker_init_fn=seed_worker # Add worker seeding
)

# ========== MODEL ==========
print(f"Loading model: {MODEL_NAME}")
model = timm.create_model(MODEL_NAME, pretrained=True)

# Modify classifier for binary output
num_ftrs = model.get_classifier().in_features
model.reset_classifier(num_classes=1) # Use reset_classifier for timm models

model.to(DEVICE)

# ========== LOSS, OPTIMIZER, SCHEDULER ==========
# Consider Focal Loss or keep BCEWithLogitsLoss (without pos_weight due to sampler)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01) # Cosine annealing from LR to LR*0.01
scaler = GradScaler() # For Mixed Precision

# ========== TRAINING & VALIDATION ==========
best_val_f1 = 0.0
best_threshold = 0.5 # Initialize best threshold

print("Starting Training...")
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0.0
    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

    for imgs, labels in train_loop:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * imgs.size(0)
        train_loop.set_postfix(loss=loss.item())

    scheduler.step() # Step the scheduler each epoch

    avg_train_loss = train_loss / len(train_loader.dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_true = []
    val_probs = []
    val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)

    with torch.no_grad():
        for imgs, labels in val_loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.float().unsqueeze(1).to(DEVICE, non_blocking=True)

            with autocast():
                 outputs = model(imgs)
                 loss = criterion(outputs, labels)

            val_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            val_probs.extend(probs.flatten())
            val_true.extend(labels.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_probs = np.array(val_probs)
    val_true = np.array(val_true)

    # --- Find Best Threshold & F1 on Validation Set ---
    current_best_f1_epoch = 0
    current_best_thresh_epoch = 0.5
    # Sort thresholds deterministically if probs have duplicates
    threshold_candidates = sorted(np.unique(val_probs), reverse=True)

    for t in threshold_candidates:
         if t == 0 or t == 1: continue # Avoid edge cases where F1 is undefined
         preds = (val_probs >= t).astype(int)
         f1 = f1_score(val_true, preds)
         if f1 > current_best_f1_epoch:
             current_best_f1_epoch = f1
             current_best_thresh_epoch = t
         # Optional deterministic tie-breaking (e.g., closer to 0.5)
         elif f1 == current_best_f1_epoch:
              if abs(t - 0.5) < abs(current_best_thresh_epoch - 0.5):
                  current_best_thresh_epoch = t
              # Further tie-breaking if still equal: prefer smaller threshold (arbitrary but deterministic)
              elif abs(t - 0.5) == abs(current_best_thresh_epoch - 0.5) and t < current_best_thresh_epoch:
                  current_best_thresh_epoch = t


    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.2f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {current_best_f1_epoch:.4f} @ Threshold: {current_best_thresh_epoch:.4f}")

    # --- Save Best Model ---
    if current_best_f1_epoch > best_val_f1:
        best_val_f1 = current_best_f1_epoch
        best_threshold = current_best_thresh_epoch # Update the global best threshold
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"🚀 New Best Model Saved! Val F1: {best_val_f1:.4f}, Threshold: {best_threshold:.4f}")
    # Deterministic tie-breaking for saving best model if F1 is equal
    elif current_best_f1_epoch == best_val_f1:
        # Optional: Save if the threshold is closer to 0.5 (or another deterministic rule)
        if abs(current_best_thresh_epoch - 0.5) < abs(best_threshold - 0.5):
             best_threshold = current_best_thresh_epoch
             torch.save(model.state_dict(), MODEL_SAVE_PATH)
             print(f"💾 Best Model Updated (Same F1, Better Threshold)! Val F1: {best_val_f1:.4f}, Threshold: {best_threshold:.4f}")
        elif abs(current_best_thresh_epoch - 0.5) == abs(best_threshold - 0.5) and current_best_thresh_epoch < best_threshold:
             best_threshold = current_best_thresh_epoch
             torch.save(model.state_dict(), MODEL_SAVE_PATH)
             print(f"💾 Best Model Updated (Same F1, Better Threshold)! Val F1: {best_val_f1:.4f}, Threshold: {best_threshold:.4f}")


training_time = time.time() - start_time
print(f"\nTraining Finished. Total Time: {training_time // 60:.0f}m {training_time % 60:.0f}s")
print(f"Best Validation F1 Score: {best_val_f1:.4f} achieved with threshold {best_threshold:.4f}")

# ========== PREDICTION (with TTA) ==========
print("\nStarting Prediction on Test Set with TTA...")
# Load best model
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(DEVICE)
model.eval()


all_tta_probs = []

# TTA Loop
for tta_idx, tta_transform in enumerate(tta_transforms):
    print(f"  Running TTA Pass {tta_idx + 1}/{len(tta_transforms)}...")
    tta_dataset = PassportDataset(TEST_CSV, TEST_IMG_DIR, transform=tta_transform, is_test=True)
    tta_loader = DataLoader(
        tta_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False, # Important: Keep shuffle=False for test/TTA
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker # Add worker seeding
    )

    current_tta_probs = []
    with torch.no_grad():
        for imgs in tqdm(tta_loader, desc=f"    Predicting (TTA {tta_idx+1})", leave=False):
            imgs = imgs.to(DEVICE, non_blocking=True)
            with autocast():
                outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            current_tta_probs.extend(probs.flatten())
    all_tta_probs.append(np.array(current_tta_probs))

# Average TTA probabilities
avg_test_probs = np.mean(all_tta_probs, axis=0)

# Apply best threshold found during validation
test_preds = (avg_test_probs >= best_threshold).astype(int)

# ========== SAVE SUBMISSION ==========
test_df = pd.read_csv(TEST_CSV)
# Ensure consistent column order if 'label' might already exist
if 'label' in test_df.columns:
    test_df = test_df.drop(columns=['label'])
test_df['label'] = test_preds
# Select only necessary columns for submission if needed (e.g., 'path', 'label')
# submission_df = test_df[['path', 'label']] # Adjust if needed
submission_df = test_df # Keep all columns for now

submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"\n✅ Submission file saved to: {SUBMISSION_PATH}")
print(f"Predicted {sum(test_preds)} positive labels out of {len(test_preds)} test samples.")