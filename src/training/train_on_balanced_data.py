# In src/training/train_on_balanced_data.py

from ultralytics import YOLO
from pathlib import Path

def train_balanced_model():
    """
    Trains the champion model (YOLOv8l) on the new, balanced dataset,
    using strong augmentation including copy-paste to further address
    class imbalance.
    """
    # 1. Configuration
    project_root = Path(__file__).resolve().parent.parent.parent
    data_yaml_path = project_root / 'data' / 'balanced_dataset' / 'balanced_dataset.yaml'
    
    if not data_yaml_path.exists():
        print(f"❌ ERROR: Balanced dataset YAML not found at {data_yaml_path}")
        print("       Please run 'create_balanced_split.py' first.")
        return

    # 2. Initialize the champion model
    model = YOLO('yolov8l.pt')

    # 3. Start the Training Process
    print(f"Starting training on balanced dataset: {data_yaml_path.name}")
    model.train(
        data=str(data_yaml_path),
        epochs=100,  # A solid number of epochs for this new dataset
        patience=30, # Stop if no improvement after 30 epochs
        batch=8,
        imgsz=640,
        project=str(project_root / 'runs' / 'training'),
        name='yolov8l_balanced_data_run',
        
        # --- Use a stable learning rate ---
        optimizer='AdamW',
        lr0=0.002,
        
        # --- KEY: Strong Augmentation for Balancing and Overfit Prevention ---
        copy_paste=0.3, # Enable copy-paste with a 30% probability per batch
        mixup=0.1,      # Use mixup for regularization
        degrees=20.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
    )
    
    print("\n✅ Training on balanced data complete!")

if __name__ == '__main__':
    train_balanced_model()