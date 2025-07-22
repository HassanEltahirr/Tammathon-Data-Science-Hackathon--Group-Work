import os
import shutil
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from PIL import Image
import imagehash
import sys

# CPU_COUNT = cpu_count()
CPU_COUNT = 10
folder = "/Temp/merge/train"

def find_image_files(start_folder):
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    file_list = []
    for root, _, files in os.walk(start_folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                path = os.path.join(root, file)
                file_list.append(path)
    return file_list

def compute_image_hash(args):
    
    file_path, hash_size = args
    try:
        with Image.open(file_path) as img:
            return (
                str(imagehash.average_hash(img, hash_size)),
                file_path
            )
    except Exception as e:
        print(f"Skipping {file_path} due to error: {str(e)}")
        return (None, file_path)

def group_by_perceptual_hash(file_list, hash_size=8):
    
    hash_groups = defaultdict(list)
    
    # Create pool of workers
    with Pool(processes=CPU_COUNT) as pool:
        results = pool.imap(
            compute_image_hash,
            [(path, hash_size) for path in file_list],
            chunksize=100
        )
        
        for hash_val, file_path in results:
            if hash_val is not None:
                hash_groups[hash_val].append(file_path)
                
    return hash_groups

def select_target_folder(file_paths):
    
    folder_counts = {}
    for path in file_paths:
        folder = os.path.dirname(path)
        if folder not in folder_counts:
            try:
                count = len(os.listdir(folder))
            except OSError:
                count = 0
            folder_counts[folder] = count
    if not folder_counts:
        return None
    max_count = max(folder_counts.values())
    return next(f for f in folder_counts if folder_counts[f] == max_count)

def move_to_target(source_path, target_folder):
    
    filename = os.path.basename(source_path)
    base, ext = os.path.splitext(filename)
    counter = 1
    while True:
        new_filename = f"{base}_{counter}{ext}" if counter > 1 else filename
        target_path = os.path.join(target_folder, new_filename)
        if not os.path.exists(target_path):
            shutil.move(source_path, target_path)
            return target_path
        counter += 1

def rename_target_folder(target_folder, source_names):
    
    base_name = os.path.basename(target_folder)
    parent_dir = os.path.dirname(target_folder)
    
    parts = base_name.split(" ")
    existing = set(parts[1:]) if len(parts) > 1 else set()
    all_duplicates = sorted(existing.union(source_names))
    
    new_name = parts[0]
    if all_duplicates:
        new_name += " " + all_duplicates[0]
    if len(all_duplicates) > 1:
        new_name += " " + all_duplicates[1]

    
    new_path = os.path.join(parent_dir, new_name)
    if new_path == target_folder:
        return target_folder
    
    try:
        os.rename(target_folder, new_path)
        print(f"Renamed folder: {target_folder} -> {new_path}")
        # Move all files from source folders to the new folder
        for name in all_duplicates[1:]:
            source_path = os.path.join(parent_dir, name)
            if os.path.exists(source_path):
                for file in os.listdir(source_path):
                    move_to_target(os.path.join(source_path, file), new_path)
                os.rmdir(source_path)  # Remove empty source folder
        return new_path
    except OSError as e:
        print(f"Error renaming folder: {e}")
        return target_folder

def process_image_duplicates(start_folder):
    
    image_files = find_image_files(start_folder)
    hash_groups = group_by_perceptual_hash(image_files)
    folder_contributions = defaultdict(set)

    for hash_key, group in hash_groups.items():
        if len(group) < 2:
            continue

        target_folder = select_target_folder(group)
        if not target_folder:
            continue

        keep_file = None
        files_to_delete = []

        for file_path in group:
            file_dir = os.path.dirname(file_path)
            if file_dir == target_folder:
                if keep_file is None:
                    keep_file = file_path
                    print(f"Keeping file: {file_path}")
                else:
                    files_to_delete.append(file_path)
            else:
                try:
                    new_path = move_to_target(file_path, target_folder)
                    if keep_file is None:
                        keep_file = new_path
                        print(f"Keeping file: {file_path}")
                    else:
                        print(f"Duplicate in target folder: {file_path}")
                        files_to_delete.append(new_path)
                except (IOError, OSError, shutil.Error):
                    continue

        for path in files_to_delete:
            try:
                os.remove(path)
                print(f"Deleted duplicate: {path}")
            except Exception as e:
                print(f"Error deleting {path}: {e}")

        source_folders = {os.path.dirname(p) for p in group}
        source_names = {os.path.basename(f) for f in source_folders if f != target_folder}
        folder_contributions[target_folder].update(source_names)

    for target_folder, names in folder_contributions.items():
        if os.path.exists(target_folder):
            rename_target_folder(target_folder, names)
        else:
            print(f"Skipping non-existent folder: {target_folder}")

if __name__ == "__main__":
    
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory")
        sys.exit(1)
    
    process_image_duplicates(folder)