# In src/testing/generate_comparison_plot.py (Version 2 - Corrected Labels)

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_comparison_plot():
    """
    Generates a grouped bar chart to compare the per-class AP@50 scores
    of the final Champion Model vs. the experimental Balanced Data Model.
    """
    # 1. The Final, Official Test Data
    class_names = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator', 'Spec.bag']
    
    # Scores from the champion YOLOv8l model trained on the original, imbalanced data
    champion_scores = [0.496, 0.447, 0.538, 0.242, 0.478, 0.106, 0.492]
    
    # Scores from the model trained on the balanced dataset (the experiment)
    balanced_scores = [0.702, 0.693, 0.822, 0.510, 0.745, 0.151, 0.601]

    # 2. Setup the Plot
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

    # --- THE FIX: Corrected the labels in the legend ---
    rects1 = ax.bar(x - width/2, champion_scores, width, label='Champion Model (Trained on Original Data)', color='royalblue')
    rects2 = ax.bar(x + width/2, balanced_scores, width, label='Experimental Model (Trained on Balanced Data)', color='darkorange')
    # ---------------------------------------------------

    # 4. Add Labels, Title, and Formatting
    ax.set_ylabel('AP@50 Score', fontsize=12)
    ax.set_title('Final Model Performance Comparison on Test Set', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.0)
    
    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()

    # 5. Save the Figure
    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = project_root / 'results' / 'Final_Model_Comparison.png'
    output_path.parent.mkdir(exist_ok=True)
    
    plt.savefig(output_path)
    
    print(f"\n✅ Comparison plot successfully generated!")
    print(f"   The plot has been saved to: {output_path}")

    # Display the plot
    plt.show()

if __name__ == '__main__':
    create_comparison_plot()