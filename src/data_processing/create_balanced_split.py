# In src/data_processing/create_balanced_split.py

import shutil
from pathlib import Path
from tqdm import tqdm
import yaml
from collections import Counter
import random

def create_balanced_dataset():
    """
    Creates a new, more balanced dataset from the final_dataset.
    - Copies val and test sets directly.
    - Undersamples the training set by enforcing a max instance count per class.
    """
    # 1. Configuration
    project_root = Path(__file__).resolve().parent.parent.parent
    source_path = project_root / 'data' / 'final_dataset'
    output_path = project_root / 'data' / 'balanced_dataset'
    
    CLASS_NAMES = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'Spec.bag']
    CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
    
    # This is the maximum number of instances any single class can have in the new training set.
    # Set this value based on your analysis to control the level of undersampling.
    # A good starting point is just above the count of your 3rd or 4th most common class.
    INSTANCE_CEILING = 4000

    # 2. Setup Directories
    print(f"Creating balanced dataset folder at: {output_path}")
    if output_path.exists(): shutil.rmtree(output_path)
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 3. Copy Validation and Test sets directly without changes
    print("Copying validation and test sets...")
    for split in ['val', 'test']:
        shutil.copytree(source_path / 'images' / split, output_path / 'images' / split, dirs_exist_ok=True)
        shutil.copytree(source_path / 'labels' / split, output_path / 'labels' / split, dirs_exist_ok=True)
    print("Validation and test sets copied.")

    # 4. Create the new, undersampled training set
    print(f"\nCreating new training set with an instance ceiling of {INSTANCE_CEILING} per class...")
    source_train_labels = list((source_path / 'labels' / 'train').glob('*.txt'))
    
    # Shuffle the list to ensure the selection is random
    random.shuffle(source_train_labels)
    
    class_counts = Counter()
    frames_copied = 0
    
    for label_path in tqdm(source_train_labels, desc="Undersampling training set"):
        # First, check what classes are in this frame
        classes_in_frame = Counter()
        with open(label_path, 'r') as f:
            for line in f:
                try:
                    class_id = int(line.split()[0])
                    classes_in_frame[CLASS_ID_TO_NAME[class_id]] += 1
                except (ValueError, IndexError):
                    continue

        # Decide if we can add this frame
        can_add_frame = True
        for class_name, count in classes_in_frame.items():
            if class_counts[class_name] + count > INSTANCE_CEILING:
                can_add_frame = False
                break
        
        # If we can add it, copy the files and update the counts
        if can_add_frame:
            # Update main counter
            class_counts.update(classes_in_frame)
            
            # Copy image file
            img_path = source_path / 'images' / 'train' / f"{label_path.stem}.png"
            if img_path.exists():
                shutil.copy(img_path, output_path / 'images' / 'train' / img_path.name)
                # Copy label file
                shutil.copy(label_path, output_path / 'labels' / 'train' / label_path.name)
                frames_copied += 1

    print(f"\nUndersampling complete. Created a new training set with {frames_copied} images.")
    print("Final training set instance counts:")
    for name in CLASS_NAMES:
        print(f"  - {name:<12}: {class_counts[name]}")

    # 5. Create the YAML file for the new balanced dataset
    print("\nCreating 'balanced_dataset.yaml' file...")
    yaml_data = {
        'path': str(output_path.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': CLASS_NAMES
    }
    yaml_filepath = output_path / 'balanced_dataset.yaml'
    with open(yaml_filepath, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    
    print("✅ Balanced dataset is ready for training.")

if __name__ == '__main__':
    create_balanced_dataset()