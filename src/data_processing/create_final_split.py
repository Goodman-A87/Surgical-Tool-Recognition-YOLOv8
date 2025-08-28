# In src/data_processing/create_final_split.py (Version 2 - Corrected)

import shutil
from pathlib import Path
from tqdm import tqdm
import yaml


def create_final_dataset_split():
    """
    Splits the consolidated 25-video dataset into strategic train, val,
    and test sets with corrected file searching logic.
    """
    # 1. Configuration
    project_root = Path(__file__).resolve().parent.parent.parent
    source_path = project_root / "data" / "consolidate_25_videos"
    output_path = project_root / "data" / "final_dataset"

    CLASS_NAMES = [
        "Grasper",
        "Bipolar",
        "Hook",
        "Scissors",
        "Clipper",
        "Irrigator",
        "Spec.bag",
    ]

    SPLIT_MAP = {
        "train": [
            "VID01",
            "VID02",
            "VID06",
            "VID07",
            "VID11",
            "VID17",
            "VID23",
            "VID31",
            "VID39",
            "VID68",
            "VID74",
            "VID92",
        ],
        "val": ["VID04", "VID37", "VID96"],
        "test": [
            "VID12",
            "VID13",
            "VID25",
            "VID30",
            "VID70",
            "VID73",
            "VID75",
            "VID103",
            "VID110",
            "VID111",
        ],
    }

    # 2. Setup Directories
    print(f"Creating final dataset folder at: {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    print("Directories created successfully.")

    # 3. Copy files to their new destinations
    # --- THE FIX: Use a recursive glob pattern '**/*.png' to find all files ---
    print(f"Searching for images in {source_path / 'images'}...")
    source_images = list((source_path / "images").glob("**/*.png"))
    print(f"Found {len(source_images)} images to process.")

    print(f"Searching for labels in {source_path / 'labels'}...")
    source_labels = list((source_path / "labels").glob("**/*.txt"))
    print(f"Found {len(source_labels)} labels to process.")
    # -------------------------------------------------------------------------

    if not source_images:
        print(
            "❌ ERROR: No images were found. Please check the 'consolidate_25_videos/images' folder."
        )
        return

    label_lookup = {f.stem: f for f in source_labels}

    print("\nSplitting files into train/val/test sets...")
    files_copied = 0
    for img_path in tqdm(source_images, desc="Copying files"):
        # The filename can be either VIDXX_... or just VIDXX. Handle both.
        try:
            video_name = img_path.stem.split("_")[0]
        except IndexError:
            video_name = img_path.stem

        target_split = None
        for split, videos in SPLIT_MAP.items():
            if video_name in videos:
                target_split = split
                break

        if target_split:
            # Copy image file
            shutil.copy(img_path, output_path / "images" / target_split / img_path.name)

            # Copy corresponding label file if it exists
            if img_path.stem in label_lookup:
                label_path = label_lookup[img_path.stem]
                shutil.copy(
                    label_path, output_path / "labels" / target_split / label_path.name
                )

            files_copied += 1

    print(f"\nFile splitting complete. Copied {files_copied} image/label pairs.")

    # 4. Create the final YAML file
    print("Creating 'final_dataset.yaml' file...")
    yaml_data = {
        "path": str(output_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }
    yaml_filepath = output_path / "final_dataset.yaml"
    with open(yaml_filepath, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)

    print("✅ Final dataset is ready for the model tournament!")


if __name__ == "__main__":
    create_final_dataset_split()
