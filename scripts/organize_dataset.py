import os
import shutil
import random

def create_dir_structure(base_path, categories):
    for category in categories:
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(base_path, split, category), exist_ok=True)

def collect_frames(source_dir, category):
    """Recursively collect all frame files from subdirectories."""
    frames = []
    category_path = os.path.join(source_dir, category)
    for root, _, files in os.walk(category_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Adjust extensions as needed
                frames.append(os.path.join(root, file))
    return frames

def split_data(source_dir, dest_dir, categories, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    create_dir_structure(dest_dir, categories)
    
    for category in categories:
        all_files = collect_frames(source_dir, category)
        print(f"Category: {category}, Total frames collected: {len(all_files)}")
        random.shuffle(all_files)
        
        train_end = int(len(all_files) * train_ratio)
        val_end = train_end + int(len(all_files) * val_ratio)
        
        splits = {
            'train': all_files[:train_end],
            'val': all_files[train_end:val_end],
            'test': all_files[val_end:]
        }
        
        for split, files in splits.items():
            for src_file in files:
                dest_file = os.path.join(dest_dir, split, category, os.path.basename(src_file))
                shutil.copy2(src_file, dest_file)
            print(f"Copied {len(files)} files to {split}/{category}")

if __name__ == "__main__":
    source_directory = 'data/frames'
    destination_directory = 'data/dataset'
    categories = ['male', 'female']
    
    split_data(source_directory, destination_directory, categories)
