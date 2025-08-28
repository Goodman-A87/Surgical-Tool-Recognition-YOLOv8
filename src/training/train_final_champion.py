# In src/training/train_final_champion.py
# In src/training/train_final_champion.py

from ultralytics import YOLO
from pathlib import Path

def run_final_training():
    """
    This is the definitive training run. It uses the champion model (YOLOv8l),
    the final dataset, and the optimal hyperparameters discovered by the
    'tune_champion_fast' process.
    """
    # 1. Configuration
    project_root = Path(__file__).resolve().parent.parent.parent
    data_yaml_path = project_root / 'data' / 'final_dataset' / 'final_dataset.yaml'
    
    if not data_yaml_path.exists():
        print(f"❌ ERROR: Dataset YAML file not found at {data_yaml_path}")
        return

    # 2. Initialize the Champion Model
    model = YOLO('yolov8l.pt')

    # 3. Start the Final, Optimized Training Process
    print("--- Starting Final Optimized Training Run ---")
    model.train(
        # --- Core Settings ---
        data=str(data_yaml_path),
        epochs=300,                 # Train for a long duration to ensure full convergence
        patience=75,                # Use early stopping as a safeguard against overfitting
        batch=8,
        imgsz=640,
        project=str(project_root / 'runs' / 'training'),
        name='yolov8l_final_champion_run', # The definitive run name
        
        # --- BEST HYPERPARAMETERS FROM TUNING ---
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.00777,
        lrf=0.00807,
        momentum=0.88914,
        weight_decay=0.00044,
        warmup_epochs=3.48191,
        warmup_momentum=0.8,
        
        # Loss function gains
        box=6.80069,
        cls=0.50592,
        dfl=1.7418,
        
        # Augmentation settings
        hsv_h=0.02576,
        hsv_s=0.8197,
        hsv_v=0.57418,
        degrees=15.42543,
        translate=0.10589,
        scale=0.50797,
        shear=1.91155,
        perspective=0.0,
        flipud=0.11835,
        fliplr=0.54214,
        mosaic=0.91929,
        mixup=0.1233
    )
    
    print("\n✅ Final training complete!")
    print(f"Your definitive model is saved in the 'runs/training/yolov8l_final_champion_run' folder.")

if __name__ == '__main__':
    run_final_training()