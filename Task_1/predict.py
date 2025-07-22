import torch
import torch.nn as nn
import faiss
import numpy as np
import pandas as pd
import os
import glob
import time
import multiprocessing
from functools import partial

from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import logging
from typing import Tuple, Optional, Dict, List, Any, Union
from collections import Counter
import imagehash


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 12 
HASHING_POOL_WORKERS = max(1, os.cpu_count() - 2 if os.cpu_count() else 1)


MODEL_CONFIGS = [
    {
        "name": "main",
        "model_path": "Model/cat_faces/model_epoch_30_loss_4.0920.pth", 
        "faiss_index_path": "Model/cat_Faces/cat_faces_epoch40.index", 
        "faiss_labels_path": "Model/cat_Faces/cat_labels_epoch40.npy", 
         
    },
    {
        "name": "specialized_1",
        "model_path": "Model/Set_1/best_model.pth", 
        "faiss_index_path": "Model/Set_1/cat_faces.index", 
        "faiss_labels_path": "Model/Set_1/cat_labels.npy", 
        
    },
    {
        "name": "specialized_2",
        "model_path": "Model/Set_2/best_model.pth", 
        "faiss_index_path": "Model/Set_2/cat_faces.index", 
        "faiss_labels_path": "Model/Set_2/cat_labels.npy", 
        
    },
    {
        "name": "specialized_3",
        "model_path": "Model/Set_3/best_model.pth", 
        "faiss_index_path": "Model/Set_3/cat_faces.index", 
        "faiss_labels_path": "Model/Set_3/cat_labels.npy", 
        
    },
    {
        "name": "cat_breed",
        "model_path": "Model/cat_breed/best_model.keras", 
    },
]


SUBMISSION_CSV_PATH = "Dataset/sample_submission.csv"
OUTPUT_CSV_PATH = "Dataset/ensemble_threshold_voting_submission.csv" 


TEST_DATA_ROOT = "Dataset"
TRAIN_DATA_ROOT = "Dataset/train" 


LABELS_ARE_FOLDERS = True
LABELS_NEED_PADDING = True
LABEL_FOLDER_PADDING = 6


MODEL_NAME = "efficientnet_b2"
EMBEDDING_SIZE = 512
IMAGE_SIZE = 224


BATCH_SIZE = 128
TOP_K = 3
FAISS_SEARCH_K = 10


HASH_SIZE = 8






SIMILARITY_UPVOTE_AVG_DIST_THRESHOLD = 20 

SIMILARITY_DOWNVOTE_AVG_DIST_THRESHOLD = 40 

SIMILARITY_SINGLE_MATCH_DIST_THRESHOLD = 15 

SIMILARITY_MAJORITY_DOWNVOTE_FRACTION = 0.7 




class CatEmbeddingModel(nn.Module):
    
    
    def __init__(self, model_name: str = "efficientnet_b2", embedding_size: int = 512, pretrained: bool = False): 
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        logging.debug(f"Initializing backbone {model_name} with {in_features} input features.")
        self.embedding_layer = nn.Linear(in_features, embedding_size)
        self.norm = nn.Identity()
        logging.debug(f"Model structure initialized with embedding size {embedding_size}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.embedding_layer(features)
        embeddings = self.norm(embeddings)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings




class CatSubmissionDataset(Dataset):
        
    def __init__(self, csv_path: str, data_root: str, transform: Optional[A.Compose] = None, hash_size: int = 8):
        self.csv_path = csv_path
        self.data_root = data_root
        self.transform = transform
        self.hash_size = hash_size

        try:
            self.df = pd.read_csv(csv_path)
            if 'filename' not in self.df.columns:
                raise ValueError("CSV file must contain a 'filename' column.")
            logging.info(f"Loaded submission CSV from {csv_path} with {len(self.df)} entries.")
        except FileNotFoundError:
            logging.error(f"Submission CSV file not found: {csv_path}")
            raise
        except Exception as e:
            logging.error(f"Error reading CSV file {csv_path}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Optional[str]]: 
        
        rel_img_path = self.df.iloc[idx]['filename']
        img_path = os.path.join(self.data_root, rel_img_path) if self.data_root else rel_img_path
        image_np = None
        test_hash_str: Optional[str] = None 

        try:
            image = Image.open(img_path).convert('RGB')
            image_np = np.array(image)
            try:
                test_hash = imagehash.average_hash(image, hash_size=self.hash_size)
                test_hash_str = str(test_hash)
            except Exception as hash_e:
                logging.warning(f"Could not hash test image {img_path} in DataLoader: {hash_e}")
                test_hash_str = None

        except FileNotFoundError:
            logging.warning(f"Image file not found: {img_path}. Returning black image tensor, hash=None.")
            image_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            test_hash_str = None
        except UnidentifiedImageError:
            logging.warning(f"Could not read image file {img_path}. Returning black image tensor, hash=None.")
            image_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            test_hash_str = None
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}. Returning black image tensor, hash=None.")
            image_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            test_hash_str = None

        if self.transform and image_np is not None:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            basic_transform = A.Compose([
                A.Resize(IMAGE_SIZE, IMAGE_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            if image_np is None:
                 image_np = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            image_tensor = basic_transform(image=image_np)["image"]

        return image_tensor, rel_img_path, test_hash_str


test_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])




def load_model(model_path: str, model_name: str, embedding_size: int, device: torch.device) -> nn.Module:
    
    
    logging.info(f"Loading model state_dict from: {model_path}")
    model = CatEmbeddingModel(model_name=model_name, embedding_size=embedding_size).to(device)
    is_data_parallel_needed = torch.cuda.device_count() > 1
    try:
        state_dict = torch.load(model_path, map_location=device)
        is_data_parallel_saved = all(k.startswith('module.') for k in state_dict.keys())
        if is_data_parallel_saved and not is_data_parallel_needed:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '', 1)] = v
            state_dict = new_state_dict
            logging.info(f"Removed 'module.' prefix from state_dict keys for {model_path}.")
            model.load_state_dict(state_dict)
        elif not is_data_parallel_saved and is_data_parallel_needed:
             model.load_state_dict(state_dict)
             logging.info(f"Wrapping model from {model_path} with DataParallel.")
             model = nn.DataParallel(model)
        elif is_data_parallel_saved and is_data_parallel_needed:
             logging.info(f"Wrapping model from {model_path} with DataParallel before loading.")
             model = nn.DataParallel(model)
             model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        logging.info(f"Model weights loaded successfully for {model_path}.")
        model.eval()
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}. Please check the path.")
        raise
    except RuntimeError as e:
        logging.error(f"Error loading state_dict for {model_path}: {e}")
        logging.error("Architecture mismatch (MODEL_NAME, EMBEDDING_SIZE) or DataParallel issue?")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during model loading for {model_path}: {e}")
        raise




def load_faiss_index(index_path: str, labels_path: str, expected_embedding_size: int) -> Tuple[faiss.Index, np.ndarray]:
    
    
    logging.info(f"Loading FAISS index from: {index_path}")
    try:
        index = faiss.read_index(index_path)
        logging.info(f"FAISS index loaded successfully with {index.ntotal} vectors from {index_path}.")
        if index.d != expected_embedding_size:
            raise ValueError(f"FAISS index dimension ({index.d}) from {index_path} != model embedding size ({expected_embedding_size}).")
    except FileNotFoundError:
        logging.error(f"FAISS index file not found: {index_path}")
        raise
    except Exception as e:
        logging.error(f"Could not load or process FAISS index {index_path}: {e}")
        raise
    logging.info(f"Loading FAISS labels mapping from: {labels_path}")
    try:
        gallery_labels = np.load(labels_path, allow_pickle=True)
        logging.info(f"FAISS labels mapping loaded successfully with {len(gallery_labels)} entries from {labels_path}.")
        if len(gallery_labels) != index.ntotal:
            logging.warning(f"Mismatch: FAISS index size ({index.ntotal}) vs labels array size ({len(gallery_labels)}).")
    except FileNotFoundError:
        logging.error(f"FAISS labels file not found at {labels_path}.")
        raise
    except Exception as e:
        logging.error(f"Could not load FAISS labels mapping from {labels_path}: {e}")
        raise
    return index, gallery_labels




models: Dict[str, nn.Module] = {}
faiss_indices: Dict[str, faiss.Index] = {}
faiss_labels: Dict[str, np.ndarray] = {}

def load_main_components():
    
    
    global models, faiss_indices, faiss_labels
    logging.info(">>> Starting Main Component Loading (Models & FAISS) <<<")
    try:
        for config in MODEL_CONFIGS:
            model_name = config["name"]
            logging.info(f"--- Loading resources for model: {model_name} ---")
            models[model_name] = load_model(
                model_path=config["model_path"],
                model_name=MODEL_NAME,
                embedding_size=EMBEDDING_SIZE,
                device=DEVICE
            )
            faiss_indices[model_name], faiss_labels[model_name] = load_faiss_index(
                index_path=config["faiss_index_path"],
                labels_path=config["faiss_labels_path"],
                expected_embedding_size=EMBEDDING_SIZE
            )
            logging.info(f"--- Finished loading resources for model: {model_name} ---")
        logging.info(">>> Finished Main Component Loading (Models & FAISS) <<<")
    except Exception as e:
        logging.error(f"Failed to load one or more models/indices during main loading. Exiting. Error: {e}", exc_info=True)
        exit(1)






def _hash_label_folder_worker(label_folder_path: str, hash_size: int, labels_need_padding: bool) -> Optional[Tuple[Any, List[imagehash.ImageHash]]]:
    
    label_name_str = os.path.basename(label_folder_path)
    label_key = label_name_str
    if labels_need_padding:
        try: label_key = int(label_name_str)
        except ValueError: pass
    current_label_hashes = []
    image_files = []
    try:
        image_files = glob.glob(os.path.join(label_folder_path, '*.[jJ][pP][gG]')) + \
                      glob.glob(os.path.join(label_folder_path, '*.[jJ][pP][eE][gG]')) + \
                      glob.glob(os.path.join(label_folder_path, '*.[pP][nN][gG]')) + \
                      glob.glob(os.path.join(label_folder_path, '*.[bB][mM][pP]'))
        if not image_files: return None
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img_hash = imagehash.average_hash(img, hash_size=hash_size)
                current_label_hashes.append(img_hash)
            except Exception: continue
        if current_label_hashes: return label_key, current_label_hashes
        else: return None
    except Exception: return None


def precompute_train_hashes_mp(train_data_root: str,
                               hash_size: int,
                               labels_are_folders: bool,
                               labels_need_padding: bool,
                               label_folder_padding: int,
                               num_pool_workers: int) -> Dict[Any, List[imagehash.ImageHash]]:
    
    if not labels_are_folders:
        logging.warning("Cannot precompute hashes: LABELS_ARE_FOLDERS is False.")
        return {}
    logging.info(f"--- Starting MP precomputation of training image hashes (Hash Size: {hash_size}, Workers: {num_pool_workers}) ---")
    start_time = time.time()
    train_hashes_map = {}
    total_hashed_images = 0
    label_folders_processed = 0
    tasks = []
    try:
        for item in os.listdir(train_data_root):
            label_folder_path = os.path.join(train_data_root, item)
            if os.path.isdir(label_folder_path): tasks.append(label_folder_path)
    except FileNotFoundError: logging.error(f"Training data root not found: {train_data_root}"); return {}
    except Exception as e: logging.error(f"Error listing directories in {train_data_root}: {e}"); return {}
    if not tasks: logging.warning(f"No subdirectories found in {train_data_root}."); return {}
    logging.info(f"Found {len(tasks)} potential label folders to process.")
    worker_func = partial(_hash_label_folder_worker, hash_size=hash_size, labels_need_padding=labels_need_padding)
    chunk_size = max(1, len(tasks) // (num_pool_workers * 4))
    results_count = 0; failed_folders = 0
    try:
        with multiprocessing.Pool(processes=num_pool_workers) as pool:
            with tqdm(total=len(tasks), desc="Hashing Folders") as pbar:
                for result in pool.imap_unordered(worker_func, tasks, chunksize=chunk_size):
                    if result:
                        label_key, hashes = result
                        train_hashes_map[label_key] = hashes
                        total_hashed_images += len(hashes)
                        results_count += 1
                    else: failed_folders += 1
                    pbar.update(1)
    except Exception as e: logging.error(f"Multiprocessing pool error during hashing: {e}", exc_info=True); return {}
    label_folders_processed = len(tasks)
    end_time = time.time()
    logging.info(f"--- Finished MP precomputing hashes ---")
    logging.info(f"Processed {label_folders_processed} potential label folders.")
    if failed_folders > 0: logging.warning(f"{failed_folders} folders resulted in errors or no valid hashes.")
    logging.info(f"Successfully precomputed {total_hashed_images} hashes for {len(train_hashes_map)} labels.")
    logging.info(f"Precomputation took {end_time - start_time:.2f} seconds.")
    if not train_hashes_map: logging.error("Precomputation resulted in an empty hash map!")
    return train_hashes_map





def calculate_hash_distances( 
                                test_hash_str: Optional[str],
                                candidate_label: Any,
                                train_hashes_map: Dict[Any, List[imagehash.ImageHash]],
                                hash_size: int 
                                ) -> List[int]: 
 
    test_hash: Optional[imagehash.ImageHash] = None

    
    if test_hash_str is not None:
        try:
            test_hash = imagehash.hex_to_hash(test_hash_str)
        except (TypeError, ValueError) as e:
            logging.warning(f"Could not convert test hash string '{test_hash_str}' back to ImageHash: {e}")
            test_hash = None

    
    if test_hash is None:
        logging.debug(f"Invalid or missing test hash string for label {candidate_label}, returning empty distance list.")
        return [] 

    precomputed_train_hashes = train_hashes_map.get(candidate_label, [])

    if not precomputed_train_hashes:
        logging.debug(f"No precomputed train hashes found for label: {candidate_label}")
        return [] 

    distances = []
    for train_hash in precomputed_train_hashes:
        try:
            
            distance = test_hash - train_hash
            distances.append(distance)
        except Exception as e:
            logging.warning(f"Error comparing hashes for label {candidate_label}: {e}")
            
            continue

    
    return distances





def generate_ensemble_predictions_with_similarity(
                                  models: Dict[str, nn.Module],
                                  faiss_indices: Dict[str, faiss.Index],
                                  faiss_labels: Dict[str, np.ndarray],
                                  train_hashes_map: Dict[Any, List[imagehash.ImageHash]],
                                  model_configs: List[Dict[str, Any]],
                                  submission_csv_path: str,
                                  test_data_root: str,
                                  output_csv_path: str,
                                  transform: A.Compose,
                                  batch_size: int,
                                  faiss_search_k: int,
                                  final_top_k: int,
                                  device: torch.device,
                                  hash_size: int,
                                
                                
                                  labels_are_folders: bool,
                                  num_dataloader_workers: int,
                                  
                                  upvote_threshold: int,
                                  downvote_threshold: int,
                                  single_match_threshold: int,
                                  majority_downvote_fraction: float
                                  ):

    logging.info(">>> Starting Prediction Generation <<<")
    try:
        submission_dataset = CatSubmissionDataset(
            csv_path=submission_csv_path,
            data_root=test_data_root,
            transform=transform,
            hash_size=hash_size
        )
        submission_loader = DataLoader(
            submission_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_dataloader_workers,
            pin_memory=True,
        )
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to create submission dataset/loader: {e}", exc_info=True)
        return

    results = []
    model_names = [cfg["name"] for cfg in model_configs]
    can_do_similarity = bool(train_hashes_map) and labels_are_folders

    logging.info(f"Starting prediction loop for {len(submission_dataset)} images...")
    if can_do_similarity:
         logging.info(f"Threshold similarity voting enabled (Upvote <= {upvote_threshold}, Downvote >= {downvote_threshold}, Single Match <= {single_match_threshold})")
    else:
         logging.warning("Image similarity check disabled (Hashes not available or LABELS_ARE_FOLDERS=False).")

    loop_start_time = time.time()
    batches_processed = 0

    with torch.no_grad():
        for images_tensor, filenames, test_hash_strs in tqdm(submission_loader, desc="Predicting (Ensemble+ThresholdVote)"):
            batch_start_time = time.time()
            images_tensor = images_tensor.to(device)
            test_hash_strs = list(test_hash_strs)

            batch_predictions: Dict[str, List[List[Union[str, int]]]] = {fname: [] for fname in filenames}
            batch_distances: Dict[str, List[List[float]]] = {fname: [] for fname in filenames}

            
            nn_time = time.time()
            for model_name in model_names:
                current_model = models[model_name]
                current_index = faiss_indices[model_name]
                current_labels = faiss_labels[model_name]
                with torch.cuda.amp.autocast(enabled=(device == torch.device("cuda"))):
                     query_embeddings_gpu = current_model(images_tensor)
                query_embeddings = query_embeddings_gpu.cpu().numpy()
                if query_embeddings.dtype != np.float32: query_embeddings = query_embeddings.astype(np.float32)
                distances, indices = current_index.search(query_embeddings, faiss_search_k)
                for i in range(len(filenames)):
                    original_filename = filenames[i]
                    retrieved_indices = indices[i]; retrieved_distances = distances[i]
                    valid_mask = retrieved_indices != -1
                    valid_indices = retrieved_indices[valid_mask]; valid_distances = retrieved_distances[valid_mask]
                    if len(valid_indices) > 0:
                        predicted_labels = current_labels[valid_indices].tolist(); predicted_distances = valid_distances.tolist()
                        if len(predicted_labels) < faiss_search_k:
                            pad_len = faiss_search_k - len(predicted_labels)
                            predicted_labels.extend([-1] * pad_len); predicted_distances.extend([float('-inf')] * pad_len)
                        batch_predictions[original_filename].append(predicted_labels[:faiss_search_k])
                        batch_distances[original_filename].append(predicted_distances[:faiss_search_k])
                    else:
                        batch_predictions[original_filename].append([-1] * faiss_search_k); batch_distances[original_filename].append([float('-inf')] * faiss_search_k)
            logging.debug(f"Batch NN+FAISS time: {time.time() - nn_time:.4f}s")


            
            combine_time = time.time()
            for i in range(len(filenames)):
                filename = filenames[i]
                test_hash_str = test_hash_strs[i]
                all_model_preds = batch_predictions[filename]
                all_model_dists = batch_distances[filename]

                
                label_nn_votes = Counter()
                label_max_dist: Dict[Union[str, int], float] = {}
                for model_idx in range(len(all_model_preds)):
                    preds = all_model_preds[model_idx]; dists = all_model_dists[model_idx]
                    for label, dist in zip(preds, dists):
                        if label != -1:
                            label_nn_votes[label] += 1 
                            label_max_dist[label] = max(label_max_dist.get(label, float('-inf')), dist)

                
                final_scores: Dict[Union[str, int], int] = {} 

                sim_check_time = time.time()
                for label, nn_vote_count in label_nn_votes.items():
                    similarity_vote = 0 

                    if can_do_similarity and test_hash_str is not None:
                        
                        all_distances = calculate_hash_distances(
                            test_hash_str=test_hash_str,
                            candidate_label=label,
                            train_hashes_map=train_hashes_map,
                            hash_size=hash_size
                        )

                        if all_distances: 
                            
                            if any(d <= single_match_threshold for d in all_distances):
                                similarity_vote = 1 
                                logging.debug(f"Sim Vote Override (+1): Single match for {filename}, Label {label}")
                            else:
                                
                                num_dissimilar = sum(1 for d in all_distances if d >= downvote_threshold)
                                if (num_dissimilar / len(all_distances)) >= majority_downvote_fraction:
                                     similarity_vote = -1 
                                     logging.debug(f"Sim Vote Override (-1): Majority mismatch for {filename}, Label {label} ({num_dissimilar}/{len(all_distances)})")
                                else:
                                     
                                     avg_dist = np.mean(all_distances)
                                     if avg_dist <= upvote_threshold:
                                         similarity_vote = 1
                                         logging.debug(f"Sim Vote (+1): Avg dist {avg_dist:.2f} <= {upvote_threshold} for {filename}, Label {label}")
                                     elif avg_dist >= downvote_threshold:
                                         similarity_vote = -1
                                         logging.debug(f"Sim Vote (-1): Avg dist {avg_dist:.2f} >= {downvote_threshold} for {filename}, Label {label}")
                                     

                    
                    final_scores[label] = nn_vote_count + similarity_vote

                logging.debug(f"Image sim vote time: {time.time() - sim_check_time:.4f}s")


                
                sort_time = time.time()
                
                sorted_candidates = sorted(
                    final_scores.items(),
                    key=lambda item: (item[1], label_max_dist.get(item[0], float('-inf'))),
                    reverse=True
                )

                
                final_labels = [label for label, score in sorted_candidates[:final_top_k]]
                if len(final_labels) < final_top_k:
                    padding = [-1] * (final_top_k - len(final_labels))
                    final_labels.extend(padding)
                results.append((filename, *final_labels))
                logging.debug(f"Image sort/select time: {time.time() - sort_time:.4f}s")

            logging.debug(f"Batch combine time: {time.time() - combine_time:.4f}s")
            batch_time = time.time() - batch_start_time
            logging.debug(f"Total batch processing time: {batch_time:.4f}s")
            batches_processed += 1


    loop_end_time = time.time()
    total_loop_time = loop_end_time - loop_start_time
    logging.info(f"Finished prediction loop. Processed {batches_processed} batches.")
    if batches_processed > 0:
        logging.info(f"Average time per batch: {total_loop_time / batches_processed:.4f} seconds.")
    logging.info(">>> Finished Prediction Generation <<<")


    
    logging.info("Saving results to CSV...")
    
    results_df = pd.DataFrame(results, columns=['filename'] + [f'label_{i+1}' for i in range(final_top_k)])
    logging.info(f"Generated {len(results_df)} final ensemble predictions.")
    try:
        original_df = pd.read_csv(submission_csv_path)
    except Exception as e:
        logging.error(f"Failed to reload original submission CSV {submission_csv_path} for merging: {e}")
        results_df.to_csv(output_csv_path, index=False)
        logging.info(f"Saved predictions directly to {output_csv_path} due to merge error.")
        return
    cols_to_drop = [f'label_{i+1}' for i in range(final_top_k)]
    original_df = original_df.drop(columns=[col for col in cols_to_drop if col in original_df.columns], errors='ignore')
    updated_df = pd.merge(original_df, results_df, on='filename', how='left')
    label_cols = [f'label_{i+1}' for i in range(final_top_k)]
    for col in label_cols:
        if col in updated_df.columns:
            updated_df[col] = updated_df[col].fillna(-1)
    try:
        updated_df.to_csv(output_csv_path, index=False)
        logging.info(f"Successfully saved updated ensemble submission file to: {output_csv_path}")
    except Exception as e:
        logging.error(f"Failed to save updated CSV to {output_csv_path}: {e}")





if __name__ == "__main__":
    logging.info("Script execution started.")
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logging.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        logging.info("Multiprocessing start method already set or not applicable.")
        pass

    if DEVICE == torch.device("cuda"):
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(DEVICE)}")
    else:
        logging.warning("Running prediction generation on CPU.")

    load_main_components()

    train_hashes_map = precompute_train_hashes_mp(
        train_data_root=TRAIN_DATA_ROOT,
        hash_size=HASH_SIZE,
        labels_are_folders=LABELS_ARE_FOLDERS,
        labels_need_padding=LABELS_NEED_PADDING,
        label_folder_padding=LABEL_FOLDER_PADDING,
        num_pool_workers=HASHING_POOL_WORKERS
    )

    if not train_hashes_map and LABELS_ARE_FOLDERS:
         logging.error("Hash precomputation failed/empty map. Similarity checks will be skipped.")

    generate_ensemble_predictions_with_similarity(
        models=models,
        faiss_indices=faiss_indices,
        faiss_labels=faiss_labels,
        train_hashes_map=train_hashes_map,
        model_configs=MODEL_CONFIGS,
        submission_csv_path=SUBMISSION_CSV_PATH,
        test_data_root=TEST_DATA_ROOT,
        output_csv_path=OUTPUT_CSV_PATH,
        transform=test_transform,
        batch_size=BATCH_SIZE,
        faiss_search_k=FAISS_SEARCH_K,
        final_top_k=TOP_K,
        device=DEVICE,
        hash_size=HASH_SIZE,
        
        
        labels_are_folders=LABELS_ARE_FOLDERS,
        num_dataloader_workers=NUM_WORKERS,
        
        upvote_threshold=SIMILARITY_UPVOTE_AVG_DIST_THRESHOLD,
        downvote_threshold=SIMILARITY_DOWNVOTE_AVG_DIST_THRESHOLD,
        single_match_threshold=SIMILARITY_SINGLE_MATCH_DIST_THRESHOLD,
        majority_downvote_fraction=SIMILARITY_MAJORITY_DOWNVOTE_FRACTION
    )

    logging.info("Script execution finished.")

