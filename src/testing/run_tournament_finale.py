# In src/testing/run_tournament_finale.py

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import time

def get_training_time(run_folder: Path) -> float:
    """Parses the results.csv to get the total training time in hours."""
    try:
        results_csv = run_folder / 'results.csv'
        if not results_csv.exists():
            return -1.0
        df = pd.read_csv(results_csv)
        # The 'time' column is cumulative, so the last value is the total time in seconds
        total_seconds = df['time'].iloc[-1]
        return total_seconds / 3600.0  # Convert to hours
    except Exception:
        return -1.0

def run_and_compare_all_variants():
    """
    Tests all 5 trained YOLOv8 tournament models, saves their results,
    generates comparison plots, and recommends a champion.
    """
    # 1. Configuration
    project_root = Path(__file__).resolve().parent.parent.parent
    data_yaml_path = project_root / 'data' / 'final_dataset' / 'final_dataset.yaml'
    tournament_runs_path = project_root / 'runs' / 'tournament'
    
    variants_to_test = ['n', 's', 'm', 'l', 'x']
    results_data = []

    print("--- Starting Tournament Finale: Testing All Models ---")

    # 2. Loop through each model variant, test it, and collect results
    for variant in variants_to_test:
        run_name = f'yolov8{variant}_50epochs'
        model_path = tournament_runs_path / run_name / 'weights' / 'best.pt'
        
        print(f"\n--- Testing YOLOv8{variant} ---")
        if not model_path.exists():
            print(f"  ❌ WARNING: Model not found at {model_path}. Skipping.")
            continue
            
        model = YOLO(model_path)
        
        metrics = model.val(
            data=str(data_yaml_path),
            split='test',
            project=str(project_root / 'runs' / 'tournament_testing'),
            name=f'test_{run_name}',
            batch=8,
            exist_ok=True,
            save_json=True # Important for getting precise AP values
        )

        # Collect the results
        training_time = get_training_time(tournament_runs_path / run_name)
        per_class_maps = metrics.box.maps
        
        result_entry = {
            'Model': f'YOLOv8{variant}',
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'Train Time (hrs)': training_time
        }
        for i, ap in enumerate(per_class_maps):
            class_name = model.names[i]
            result_entry[class_name] = ap.item()
        
        results_data.append(result_entry)
        print(f"  ✅ Testing complete for YOLOv8{variant}. mAP50: {metrics.box.map50:.4f}")

    if not results_data:
        print("\n❌ FATAL ERROR: No models were successfully tested. Aborting.")
        return

    # 3. Create a pandas DataFrame for easy analysis and plotting
    df = pd.DataFrame(results_data).set_index('Model')
    
    # --- 4. Generate and Save Plots ---
    print("\n--- Generating Comparison Plots ---")
    
    # Plot 1: Overall mAP50 Comparison
    plt.figure(figsize=(10, 6))
    df['mAP50'].sort_values().plot(kind='barh', color='skyblue')
    plt.title('Overall mAP@50 Performance of YOLOv8 Variants')
    plt.xlabel('mAP@50 Score')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot1_path = tournament_runs_path / 'tournament_overall_map_comparison.png'
    plt.savefig(plot1_path)
    print(f"  - Saved overall performance plot to {plot1_path}")

    # Plot 2: Per-Class mAP50 Comparison
    class_names = [name for name in df.columns if name not in ['mAP50', 'mAP50-95', 'Train Time (hrs)']]
    df[class_names].T.plot(kind='bar', figsize=(15, 8), width=0.8)
    plt.title('Per-Class mAP@50 Performance')
    plt.ylabel('mAP@50 Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model Variant')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot2_path = tournament_runs_path / 'tournament_per_class_map_comparison.png'
    plt.savefig(plot2_path)
    print(f"  - Saved per-class performance plot to {plot2_path}")
    
    # --- 5. Recommend a Champion ---
    print("\n--- Tournament Results Summary ---")
    print(df.round(3)) # Print the full results table
    
    # Simple logic for recommendation: highest mAP50
    champion = df['mAP50'].idxmax()
    print("\n--- Champion Recommendation ---")
    print(f"🏆 The champion model is: {champion}")
    print("   This model achieved the highest mAP@50 score on the test set.")
    print("   Consider this model for your intensive optimization experiments.")

if __name__ == '__main__':
    run_and_compare_all_variants()