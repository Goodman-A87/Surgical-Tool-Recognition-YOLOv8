# python src\training\run_tournament.py

from ultralytics import YOLO
from pathlib import Path
import argparse
import torch

def run_training_for_model(model_variant: str): 
    """
    Trains a specific YOLOv8 model variant on our final_dataset.

    Args:
        model_variant (str): The YOLOv8 variant to train (e.g., 'n', 's', 'm', 'l', 'x').
    """
    # 1. Configuration
    project_root = Path(__file__).resolve().parent.parent.parent
    data_yaml_path = project_root / 'data' / 'final_dataset' / 'final_dataset.yaml'
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- Starting Tournament Round for YOLOv8{model_variant} on {device} ---")
    
    if not data_yaml_path.exists():
        print(f"❌ ERROR: Dataset YAML file not found at {data_yaml_path}")
        return

    # 2. Initialize the specified YOLOv8 model
    # The model name is constructed like 'yolov8n.pt', 'yolov8s.pt', etc.
    model_name = f'yolov8{model_variant}.pt'
    print(f"Initializing model from pre-trained weights: {model_name}")
    model = YOLO(model_name)

    # 3. Start the Training Process
    # We use consistent settings for a fair comparison.
    model.train(
        data=str(data_yaml_path),
        epochs=50,                  
        batch=8,                    
        imgsz=640,
        project=str(project_root / 'runs' / 'tournament'),
        name=f'yolov8{model_variant}_50epochs', 
        exist_ok=True,              # Allows re-running the same experiment
    )
    
    print(f"\n✅ Tournament round for YOLOv8{model_variant} complete!")

if __name__ == '__main__':
    # Use argparse to accept the model variant from the command line
    parser = argparse.ArgumentParser(description="Run a YOLOv8 training tournament.")
    parser.add_argument(
        '--variant', 
        type=str, 
        required=True, 
        choices=['n', 's', 'm', 'l', 'x'],
        help="The YOLOv8 model variant to train (n, s, m, l, or x)."
    )
    
    args = parser.parse_args()
    
    try:
        import ultralytics
    except ImportError:
        print("Please install ultralytics: pip install ultralytics")
    else:
        run_training_for_model(model_variant=args.variant)