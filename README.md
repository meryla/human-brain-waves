Motor Imagery EEG Brain-Computer Interface (BCI)

An end-to-end BCI pipeline processing raw multi-channel EEG signals from the PhysioNet dataset, extracting spatial features using regularized Common Spatial Patterns (CSP), classifying motor intent (Left vs. Right Hand) via Support Vector Machines (SVM), and displaying live predictions on a dark-mode GUI.

🚀 Quick Start

Bash
# Clone & setup environment
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install mne numpy scipy scikit-learn matplotlib customtkinter

# Run pipeline or GUI
python app.py

📌 Features

Dataset: PhysioNet EEGMMIDB (64 channels, 160 Hz).

Preprocessing: 8.0–30.0 Hz bandpass filter (μ and β rhythms), cropped to t∈[0.5,3.5 s].

Channels: Restricted to 15 sensory-motor electrodes (FC3..4, C3..4, CP3..4).

Model: Ledoit-Wolf Regularized CSP (4 components) + Linear SVM.

Interface: CustomTkinter dark-mode GUI with embedded Matplotlib canvas.

📊 Results (80/20 Holdout Split)

Subject	Trials (N)	Test Accuracy	Note
Subject 1	90	72.22%	Robust generalization
Subject 2	90	94.44%	Strong sensorimotor modulation
Subject 3	90	50.00%	Demonstrates BCI Illiteracy
👤 Author

Maryam Lachhab

Master in Applied Informatics: Artificial Intelligence and Data Science

Vrije Universiteit Brussel (VUB)