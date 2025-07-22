import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_metric_learning import losses, miners, distances
import numpy as np
import os
import random
from tqdm import tqdm
import faiss
from PIL import Image, UnidentifiedImageError
import logging
from typing import Tuple, Optional, Dict, List
from multiprocessing import cpu_count

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


TRAIN_PATH = "Temp/train"
TEST_PATH = "Temp/non"
USE_VAL = False
MODEL_NAME = "efficientnet_b2"
EMBEDDING_SIZE = 512
BATCH_SIZE = 100
NUM_WORKERS = 12
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20
MARGIN = 0.3
MINER_EPSILON = 0.1
GRADIENT_CLIP_VAL = 1.0
MODEL_DIR = "Model/final/"

LOSS_TYPE = "ArcFaceLoss"
MINER_TYPE = None


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = "best_model.pth"
FINAL_MODEL_PATH = "final_model.pth"
FAISS_INDEX_PATH = "cat_faces.index"
FAISS_LABELS_PATH = "cat_labels.npy"


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CatDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[A.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")

        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not self.classes:
            raise ValueError(f"No subdirectories (classes) found in {root_dir}")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.images, self.labels = self._load_images()
        self.num_classes = len(self.classes)
        logging.info(f"Found {len(self.images)} images in {self.num_classes} classes from {root_dir}.")

    def _load_images(self) -> Tuple[List[str], List[int]]:
        images = []
        labels = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir): continue

            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    images.append(img_path)
                    labels.append(self.class_to_idx[cls])
        return images, labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)
        except UnidentifiedImageError:
            logging.warning(f"Could not read image file {img_path}, returning placeholder.")
            image_np = np.zeros((224, 224, 3), dtype=np.uint8)
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            image_np = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            basic_transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            image_tensor = basic_transform(image=image_np)["image"]

        return image_tensor, label


train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


class CatEmbeddingModel(nn.Module):
    def __init__(self, model_name: str = "efficientnet_b2", embedding_size: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        logging.info(f"Loaded backbone {model_name} with {in_features} input features.")

        self.embedding_layer = nn.Linear(in_features, embedding_size)
        self.norm = nn.Identity()

        self._initialize_weights()
        logging.info(f"Model initialized with embedding size {embedding_size}.")

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.embedding_layer.weight, mode='fan_out', nonlinearity='relu')
        if self.embedding_layer.bias is not None:
            nn.init.constant_(self.embedding_layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.embedding_layer(features)
        embeddings = self.norm(embeddings)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


def get_loss_miner(loss_type: str, miner_type: Optional[str], num_classes: int, embedding_size: int, margin: float, epsilon: float):
    distance = distances.CosineSimilarity()

    if loss_type == "TripletMarginLoss":
        loss_func = losses.TripletMarginLoss(margin=margin, distance=distance)
        logging.info(f"Using TripletMarginLoss with margin={margin} and Cosine distance.")
    elif loss_type == "ArcFaceLoss":
        loss_func = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embedding_size, margin=margin, scale=64)
        logging.info(f"Using ArcFaceLoss with margin={margin}, scale=64, num_classes={num_classes}.")
    elif loss_type == "CosFaceLoss":
        loss_func = losses.CosFaceLoss(num_classes=num_classes, embedding_size=embedding_size, margin=margin, scale=64)
        logging.info(f"Using CosFaceLoss with margin={margin}, scale=64, num_classes={num_classes}.")
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    if miner_type == "MultiSimilarityMiner":
        miner = miners.MultiSimilarityMiner(epsilon=epsilon, distance=distance)
        logging.info(f"Using MultiSimilarityMiner with epsilon={epsilon}.")
    elif miner_type is None:
        miner = None
        logging.info("No miner specified (expected for ArcFace/CosFace).")
    else:
        raise ValueError(f"Unsupported miner type: {miner_type}")

    return loss_func.to(DEVICE), miner


def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    optimizer: optim.Optimizer,
                    loss_func: nn.Module,
                    miner: Optional[miners.BaseMiner],
                    scaler: torch.cuda.amp.GradScaler,
                    epoch: int,
                    gradient_clip_val: float) -> float:
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    progress_bar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (data, labels) in progress_bar:
        data, labels = data.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            embeddings = model(data)

            if miner:
                indices_tuple = miner(embeddings, labels)
                if all(len(t) == 0 for t in indices_tuple):
                    logging.debug(f"Miner returned no pairs/triplets for batch {batch_idx}, skipping loss computation.")
                    continue
                loss = loss_func(embeddings, labels, indices_tuple)
            else:
                loss = loss_func(embeddings, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

        scaler.step(optimizer)

        scaler.update()

        current_loss = loss.item()
        total_loss += current_loss
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", avg_loss=f"{total_loss / (batch_idx + 1):.4f}")

    avg_loss = total_loss / num_batches
    logging.info(f"Epoch {epoch} [Train] Average Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model: nn.Module,
             val_loader: DataLoader,
             loss_func: nn.Module,
             miner: Optional[miners.BaseMiner],
             epoch: int) -> float:
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    progress_bar = tqdm(val_loader, total=num_batches, desc=f"Epoch {epoch} [Val]")

    with torch.no_grad():
        for data, labels in progress_bar:
            data, labels = data.to(DEVICE), labels.to(DEVICE)

            with torch.cuda.amp.autocast():
                embeddings = model(data)

                try:
                    if miner:
                        indices_tuple = miner(embeddings, labels)
                        if all(len(t) == 0 for t in indices_tuple):
                            loss = torch.tensor(0.0)
                        else:
                            loss = loss_func(embeddings, labels, indices_tuple)
                    else:
                        loss = loss_func(embeddings, labels)
                    current_loss = loss.item()
                except Exception as e:
                    logging.warning(f"Error calculating validation loss: {e}. Assigning loss=0.")
                    current_loss = 0.0

            total_loss += current_loss
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", avg_loss=f"{total_loss / (len(progress_bar)):.4f}")

    avg_loss = total_loss / num_batches
    logging.info(f"Epoch {epoch} [Val] Average Loss: {avg_loss:.4f}")
    return avg_loss


def create_faiss_index(model: nn.Module,
                       dataset: CatDataset,
                       embedding_size: int,
                       batch_size: int) -> Tuple[Optional[faiss.Index], Optional[np.ndarray]]:
    model.eval()
    index = faiss.IndexFlatIP(embedding_size)
    all_embeddings = []
    all_labels = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    logging.info(f"Starting FAISS index creation for {len(dataset)} images...")

    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Creating FAISS Index"):
            data = data.to(DEVICE)
            with torch.cuda.amp.autocast():
                embeddings = model(data).cpu().numpy()

            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            faiss.normalize_L2(embeddings)

            all_embeddings.append(embeddings)
            all_labels.extend(labels.cpu().numpy())

    if not all_embeddings:
        logging.error("No embeddings were generated. Cannot create FAISS index.")
        return None, None

    all_embeddings_np = np.vstack(all_embeddings)
    index.add(all_embeddings_np)
    logging.info(f"FAISS index created with {index.ntotal} vectors.")

    all_labels_np = np.array(all_labels)
    return index, all_labels_np


def main():
    if DEVICE == torch.device("cpu"):
        logging.warning("CUDA not available, training on CPU. This will be very slow.")

    logging.info("Loading datasets...")
    try:
        train_dataset = CatDataset(root_dir=TRAIN_PATH, transform=train_transform)
        num_classes = train_dataset.num_classes
        logging.info(f"Number of classes detected: {num_classes}")

        if USE_VAL:
            val_dataset = CatDataset(root_dir=TEST_PATH, transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
        else:
            val_loader = None

        test_dataset = CatDataset(root_dir=TEST_PATH if USE_VAL else TRAIN_PATH, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS, drop_last=True)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error initializing datasets: {e}")
        return

    logging.info("Initializing model, loss, optimizer...")
    model = CatEmbeddingModel(model_name=MODEL_NAME, embedding_size=EMBEDDING_SIZE).to(DEVICE)
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    loss_func, miner = get_loss_miner(LOSS_TYPE, MINER_TYPE, num_classes, EMBEDDING_SIZE, MARGIN, MINER_EPSILON)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE / 100)

    scaler = torch.cuda.amp.GradScaler()

    logging.info("Starting training...")
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, miner, scaler, epoch, GRADIENT_CLIP_VAL)

        if USE_VAL and val_loader:
            val_loss = validate(model, val_loader, loss_func, miner, epoch)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), MODEL_DIR + MODEL_SAVE_PATH)
                logging.info(f"New best model saved at epoch {epoch} with validation loss {val_loss:.4f}")
        else:
            scheduler.step()

            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_epoch = epoch
                torch.save(model.state_dict(), MODEL_DIR + MODEL_SAVE_PATH)
                logging.info(f"New best model saved at epoch {epoch} with training loss {train_loss:.4f}")

            if epoch % 5 == 0:
                torch.save(model.state_dict(), MODEL_DIR + f"model_epoch_{epoch}_loss_{train_loss:.4f}.pth")
                logging.info(f"Model saved at epoch {epoch} with training loss {train_loss:.4f}")
            pass

    logging.info("Training finished.")
    torch.save(model.state_dict(), MODEL_DIR + FINAL_MODEL_PATH)
    logging.info(f"Final model saved to {MODEL_DIR + FINAL_MODEL_PATH}")
    if USE_VAL:
        logging.info(f"Best model was saved at epoch {best_epoch} with validation loss {best_val_loss:.4f}")

    logging.info("Creating FAISS index for the training set...")
    if USE_VAL and os.path.exists(MODEL_DIR + MODEL_SAVE_PATH):
        logging.info(f"Loading best model from {MODEL_DIR + MODEL_SAVE_PATH} for indexing.")
        model.load_state_dict(torch.load(MODEL_DIR + MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        logging.info(f"Using final model from {FINAL_MODEL_PATH} for indexing.")

    gallery_dataset = CatDataset(root_dir=TRAIN_PATH, transform=val_transform)
    faiss_index, gallery_labels = create_faiss_index(model, gallery_dataset, EMBEDDING_SIZE, BATCH_SIZE*2)

    if faiss_index and gallery_labels is not None:
        faiss.write_index(faiss_index, MODEL_DIR + FAISS_INDEX_PATH)
        np.save(MODEL_DIR + FAISS_LABELS_PATH, gallery_labels)
        logging.info(f"FAISS index saved to {MODEL_DIR + FAISS_INDEX_PATH}")
        logging.info(f"FAISS labels saved to {MODEL_DIR + FAISS_LABELS_PATH}")
    else:
        logging.error("FAISS index creation failed. Skipping evaluation.")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
