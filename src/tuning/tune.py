# In src/tuning/tune_champion.py

from ultralytics import YOLO
from pathlib import Path
import torch

def tune_champion_model():
    """
    Runs a revised, much faster hyperparameter tuning session on our
    champion model (YOLOv8l) to meet project deadlines.
    """
    # 1. Configuration
    project_root = Path(__file__).resolve().parent.parent.parent
    data_yaml_path = project_root / 'data' / 'final_dataset' / 'final_dataset.yaml'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Starting FAST Champion Tuning Session for YOLOv8l on {device} ---")
    
    if not data_yaml_path.exists():
        print(f"❌ ERROR: Dataset YAML file not found at {data_yaml_path}")
        return

    # 2. Initialize the Champion Model
    model = YOLO('yolov8l.pt')

    # 3. Start the Faster Tuning Process
    print("Tuning will now commence with revised settings for speed...")
    model.tune(
        # --- Core Settings ---
        data=str(data_yaml_path),
        
        # --- KEY CHANGES FOR SPEED ---
        epochs=10,          # Reduced from 30. Get a quick signal from each run.
        iterations=25,      # Reduced from 50. Run a solid number of experiments.
        # ---------------------------

        optimizer='AdamW',
        project=str(project_root / 'runs' / 'tuning'),
        name='yolov8l_champion_tune_fast', # New name for this run
        
        # --- Keep Strong Augmentation ---
        degrees=20.0,
        translate=0.1,
        scale=0.6,
        shear=2.0,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        
        plots=False,
        save=False,
        val=True
    )
    
    print("\n✅ Fast champion tuning complete!")
    print(f"Full results are saved in: {project_root / 'runs' / 'tuning' / 'yolov8l_champion_tune'}")

if __name__ == '__main__':
    tune_champion_model()