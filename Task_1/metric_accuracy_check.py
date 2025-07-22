import torch
import torch.nn as nn 
import faiss
import numpy as np
import os
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError 
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import logging
from typing import Tuple, Optional, Dict, List 




logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 11


MODEL_PATH = "Model/final/best_model.pth"       
FAISS_INDEX_PATH = "Model/final/cat_faces.index"
FAISS_LABELS_PATH = "Model/final/cat_labels.npy"     
TEST_DATA_ROOT = "Temp/test"              

MODEL_NAME = "efficientnet_b2"           
EMBEDDING_SIZE = 512                     

BATCH_SIZE = 64                         
TOP_K_VALUES = [1, 3]                




class CatEmbeddingModel(nn.Module):
    
    def __init__(self, model_name: str = "efficientnet_b2", embedding_size: int = 512, pretrained: bool = True):
        super().__init__()
        
        
        
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        logging.info(f"Initializing backbone {model_name} with {in_features} input features for loading weights.")

        
        self.embedding_layer = nn.Linear(in_features, embedding_size)
        
        
        
        self.norm = nn.Identity() 

        
        logging.info(f"Model structure initialized with embedding size {embedding_size}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.embedding_layer(features)
        embeddings = self.norm(embeddings)
        
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings




class CatTestDataset(Dataset):
    
    def __init__(self, root_dir: str, transform: Optional[A.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        if not os.path.isdir(root_dir):
             raise FileNotFoundError(f"Test data root directory not found: {root_dir}")

        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not self.classes:
            raise ValueError(f"No subdirectories (classes) found in {root_dir}")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.samples = self._load_samples()
        logging.info(f"Found {len(self.samples)} test images in {len(self.classes)} classes from {root_dir}.")

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir): continue

            for img_name in os.listdir(class_dir):
                 img_path = os.path.join(class_dir, img_name)
                 if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                     samples.append((img_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
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


test_transform = A.Compose([
    A.Resize(224, 224), 
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])




logging.info(f"Loading model state_dict from: {MODEL_PATH}")

model = CatEmbeddingModel(model_name=MODEL_NAME, embedding_size=EMBEDDING_SIZE).to(DEVICE)
if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
        
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    logging.info("Model weights loaded successfully.")
except FileNotFoundError:
     logging.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
     exit()
except RuntimeError as e:
     logging.error(f"Error loading state_dict: {e}")
     logging.error("This usually means the model architecture in this script doesn't match the saved model's architecture.")
     exit()

model.eval() 




logging.info(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    logging.info(f"FAISS index loaded successfully with {index.ntotal} vectors.")
except Exception as e:
    logging.error(f"Could not load FAISS index: {e}")
    exit()

logging.info(f"Loading FAISS labels mapping from: {FAISS_LABELS_PATH}")
try:
    
    
    gallery_labels = np.load(FAISS_LABELS_PATH)
    logging.info(f"FAISS labels mapping loaded successfully with {len(gallery_labels)} entries.")
    if len(gallery_labels) != index.ntotal:
         logging.warning(f"Mismatch between FAISS index size ({index.ntotal}) and labels array size ({len(gallery_labels)}). Ensure they correspond to the same indexing run.")
except FileNotFoundError:
    logging.error(f"FAISS labels file not found at {FAISS_LABELS_PATH}. This file is created during training.")
    exit()
except Exception as e:
    logging.error(f"Could not load FAISS labels mapping: {e}")
    exit()





def calculate_recall_at_k(k_values: List[int]) -> Dict[int, float]:
    
    
    try:
        test_dataset = CatTestDataset(TEST_DATA_ROOT, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error initializing test dataset: {e}")
        return {k: 0.0 for k in k_values} 

    total_correct = {k: 0 for k in k_values}
    total_samples = 0
    max_k = max(k_values)

    logging.info(f"Starting Recall@{max_k} evaluation...")
    with torch.no_grad():
        for images, true_labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(DEVICE)
            
            true_labels_np = true_labels.cpu().numpy()
            batch_size = len(true_labels_np)
            total_samples += batch_size

            
            with torch.cuda.amp.autocast(): 
                 query_embeddings = model(images).cpu().numpy()

            
            if query_embeddings.dtype != np.float32:
                query_embeddings = query_embeddings.astype(np.float32)

            
            
            distances, indices = index.search(query_embeddings, max_k)

            
            for i in range(batch_size): 
                query_label = true_labels_np[i] 
                retrieved_indices = indices[i] 

                
                valid_indices = retrieved_indices[retrieved_indices != -1]
                if len(valid_indices) == 0: continue 

                
                retrieved_labels = gallery_labels[valid_indices]

                for k in k_values:
                    
                    if query_label in retrieved_labels[:k]:
                        total_correct[k] += 1

    recall_results = {k: (total_correct[k] / total_samples) if total_samples > 0 else 0.0 for k in k_values}
    print("\n--- Evaluation Results ---")
    for k, recall in recall_results.items():
        print(f"Recall@{k}: {recall:.4f} ({total_correct[k]}/{total_samples})")
    print("------------------------")
    return recall_results




if __name__ == "__main__":
    if DEVICE == torch.device("cpu"):
        logging.warning("Running evaluation on CPU.")
    
    

    calculate_recall_at_k(k_values=TOP_K_VALUES)
