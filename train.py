import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from mne.decoding import CSP

from preprocess import load_and_filter_data, get_epochs

def train_bci_model(epochs):
    print("\n--- Starting Model Training Pipeline ---")
    
    # 1. Get our data arrays and labels
    # X shape: (trials, channels, timepoints)
    # y shape: (trials,) where 0 = left_hand, 1 = right_hand
    X = epochs.get_data() 
    y = epochs.events[:, -1]
    
    # Convert event codes (2 and 3) to standard 0 and 1
    # This makes interpreting predictions much simpler!
    unique_labels = np.unique(y)
    y = np.where(y == unique_labels[0], 0, 1)
    
    print(f"Data features shape (X): {X.shape}")
    print(f"Labels shape (y): {y.shape} (0: Left, 1: Right)")
    
    # 2. Split into Train (80%) and Test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Build the pipeline: CSP (Feature Extraction) -> SVM (Classifier)
    # We will extract 4 spatial patterns from the EEG channels
    csp = CSP(n_components=4, reg='ledoit_wolf', log=True, cov_est='epoch')
    svc = SVC(kernel='linear', C=1.0, probability=True)
    
    bci_pipeline = Pipeline([
        ('CSP', csp),
        ('Classifier', svc)
    ])
    
    # 4. Train the model!
    print("Fitting CSP and training the SVM classifier...")
    bci_pipeline.fit(X_train, y_train)
    
    # 5. Evaluate the model
    y_pred = bci_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Evaluation Results ---")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Left Hand', 'Right Hand']))
    
    return bci_pipeline

if __name__ == "__main__":
    # 1. Fetch & Preprocess
    raw_filtered = load_and_filter_data(subject=1, runs=[3, 4, 7, 8, 11, 12])
    epochs = get_epochs(raw_filtered)
    
    # 2. Train
    trained_pipeline = train_bci_model(epochs)