# In src/data_processing/counting_tool_appearances.py

from pathlib import Path
from collections import Counter
from tqdm import tqdm


def analyze_consolidated_dataset():
    """
    Scans the consolidated 25-video dataset to count tool instances,
    with corrected path logic to work from within 'src/data_processing'.
    """
    # 1. Configuration
    # --- THE FIX: Go up three levels instead of two to find the project root ---
    # From: .../src/data_processing/counting_tool_appearances.py
    # .parent -> .../src/data_processing/
    # .parent.parent -> .../src/
    # .parent.parent.parent -> .../ (Project Root)
    project_root = Path(__file__).resolve().parent.parent.parent
    # --------------------------------------------------------------------

    labels_base_path = project_root / "data" / "consolidated_25_videos" / "labels"

    CLASS_ID_TO_NAME = {
        0: "Grasper",
        1: "Bipolar",
        2: "Hook",
        3: "Scissors",
        4: "Clipper",
        5: "Irrigator",
        6: "Spec.bag",
    }

    if not labels_base_path.exists():
        print(f"❌ ERROR: The directory '{labels_base_path}' was not found.")
        print("       Please ensure the path is correct and the data exists.")
        return

    # 2. Data Structures
    video_counts = {}
    grand_total_counts = Counter()

    print(f"Scanning for label files in: {labels_base_path}")
    all_label_files = list(labels_base_path.glob("*.txt"))

    if not all_label_files:
        print("❌ ERROR: No .txt label files were found.")
        return

    for label_file in tqdm(all_label_files, desc="Analyzing label files"):
        video_name = label_file.stem.split("_")[0]

        if video_name not in video_counts:
            video_counts[video_name] = Counter()

        with open(label_file, "r") as f:
            for line in f:
                try:
                    class_id = int(line.split()[0])
                    if class_id in CLASS_ID_TO_NAME:
                        class_name = CLASS_ID_TO_NAME[class_id]
                        video_counts[video_name][class_name] += 1
                        grand_total_counts[class_name] += 1
                except (ValueError, IndexError):
                    continue

    # 3. Generate and Print the Report
    print("\n\n--- Tool Instance Report for Consolidated 25 Videos ---")

    # Per-Video Breakdown
    sorted_video_names = sorted(
        video_counts.keys(), key=lambda v: int(v.replace("VID", ""))
    )

    for video_name in sorted_video_names:
        counts = video_counts[video_name]
        total_in_video = sum(counts.values())
        print(f"\n--- Report for {video_name} (Total: {total_in_video} instances) ---")
        if not counts:
            print("  No tools found.")
            continue
        for tool_name, count in counts.most_common():
            print(f"  - {tool_name:<12}: {count:>6} instances")

    # Grand Total Summary
    grand_total = sum(grand_total_counts.values())
    print(f"\n\n--- Grand Total Summary (Total: {grand_total} instances) ---")
    for tool_name, count in grand_total_counts.most_common():
        percentage = (count / grand_total) * 100
        print(f"  - {tool_name:<12}: {count:>6} instances ({percentage:5.2f}%)")

    print(
        "\n✅ Analysis complete. Use this report to create your strategic train/val/test split."
    )


if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("Please install tqdm for a progress bar: pip install tqdm")
    finally:
        analyze_consolidated_dataset()
