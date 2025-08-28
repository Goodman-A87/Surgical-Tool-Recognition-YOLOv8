
Automated Recognition of Surgical Instruments using YOLOv8


Author: Abraham Goodman
Dissertation for: MSc Robotics and Automation, University of Salford

---

1. Project Overview

---

This project presents a comprehensive workflow for training and evaluating a state-of-the-art YOLOv8 model for the real-time detection of surgical instruments in laparoscopic cholecystectomy videos. The study utilizes a consolidated and strategically split dataset of 25 surgical videos to address the challenges of class imbalance and the complex intraoperative visual environment.

The project follows a rigorous methodology:

1. Data Curation: A 25-video dataset was created, cleaned, and strategically split to prioritize rare classes in the training set.
2. Model Selection: A tournament was conducted, training five YOLOv8 variants (n, s, m, l, x) to find the optimal architecture.
3. Optimization & Analysis: The champion model was subjected to advanced experiments, including hyperparameter tuning and data balancing, to fully characterize its performance.

---

2. Final Results

---

The definitive champion model, a YOLOv8l trained for 50 epochs on the original imbalanced data, achieved the highest and most robust performance on the official 10-video test set.

- Final Test Set mAP@50: 64.4%

This result is highly competitive with published benchmarks and demonstrates the model's ability to generalize to unseen surgical data.

---

3. Project Structure

---

- /src/: Contains all Python source code.
  - data_processing/: Scripts for cleaning, splitting, and preparing datasets.
  - training/: Scripts for training the models.
  - testing/: Scripts for evaluating models and generating demonstrations.
    -tuning/: Script for hyperparameter tuning.
- /data/: (Not included) This folder should contain the raw and processed datasets.
- /runs/: (Not included) This folder contains all training and testing outputs, including the final trained models.

---

4. How to Run This Project

---

## a. Setup the Environment

# 1. Clone the repository

git clone https://github.com/Goodman-A87/Surgical-Tool-Recognition-YOLOv8.git
cd surgical_Tool_Recognition

# 2. Create and activate a virtual environment

python -m venv .venv

# On Windows:

.venv\Scripts\activate

# On macOS/Linux:

# source .venv/bin/activate

# 3. Install the required libraries

pip install -r requirements.txt

## b. Obtain and Prepare the Data

The datasets (CholecTrack20, CholecT50, etc.) must be downloaded from their official sources (e.g., Synapse). Once downloaded, they should be placed in a `data/raw` directory.

The final, cleaned dataset can then be created by running:
python src/data_processing/create_final_split.py

## c. Reproduce the Main Result (Train & Test the Champion)

# 1. Train the YOLOv8l model for 50 epochs

python src/training/run_tournament.py --variant l

# 2. Test the trained model on the official test set

python src/testing/test_baseline_champion.py

---

5. Key Dependencies

---

- Python 3.10+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Pandas & Matplotlib
