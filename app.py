# 1. FORCE Matplotlib to use the Tkinter-compatible backend FIRST
import matplotlib
matplotlib.use("TkAgg")  # This must be the absolute first line!

# 2. Now import the rest of your GUI and Backend libraries
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import numpy as np

# Corrected Tkinter-specific Canvas backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Import our backend pipeline functions
from preprocess import load_and_filter_data, get_epochs
from train import train_bci_model
# Set modern dark-themed appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class BCIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("AI-Powered Brain-Computer Interface Dashboard")
        self.geometry("1000x650")
        
        # Application state variables
        self.subject = 1
        self.epochs = None
        self.model = None
        self.current_trial_idx = 0
        
        # --- UI LAYOUT ---
        # Configure grid layout (1 row, 2 columns: Left sidebar, Right visualization)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # 1. Left Sidebar (Controls)
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.setup_sidebar()
        
        # 2. Right Main Dashboard Area
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.setup_main_dashboard()
        
        # Load initial subject data on startup
        self.load_and_train_subject()

    def setup_sidebar(self):
        # App Title
        self.title_label = ctk.CTkLabel(self.sidebar_frame, text="BCI Controller", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(pady=(30, 20), padx=20)
        
        # Subject Selector Dropdown
        self.subj_label = ctk.CTkLabel(self.sidebar_frame, text="Select Volunteer (Subject):", font=ctk.CTkFont(size=14))
        self.subj_label.pack(pady=(10, 5), padx=20, anchor="w")
        
        self.subj_dropdown = ctk.CTkComboBox(self.sidebar_frame, values=["Subject 1", "Subject 2", "Subject 3"], command=self.on_subject_change)
        self.subj_dropdown.pack(pady=(0, 20), padx=20, fill="x")
        
        # Model Training Status Info
        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Initializing...", font=ctk.CTkFont(size=12), text_color="yellow")
        self.status_label.pack(pady=10, padx=20, anchor="w")
        
        # "Load Next Trial" Button
        self.next_btn = ctk.CTkButton(self.sidebar_frame, text="🔄 Load Next Brainwave Trial", command=self.load_next_trial)
        self.next_btn.pack(pady=(100, 10), padx=20, fill="x")
        
        # Big "PREDICT INTENT" Button
        self.predict_btn = ctk.CTkButton(self.sidebar_frame, text="🧠 Predict Intent", fg_color="#2ecc71", hover_color="#27ae60", font=ctk.CTkFont(size=16, weight="bold"), command=self.predict_intent)
        self.predict_btn.pack(pady=(10, 20), padx=20, fill="x")

    def setup_main_dashboard(self):
        # Header showing selected trials
        self.header_label = ctk.CTkLabel(self.main_frame, text="EEG Signal Monitor (Channels FC3, Cz, FC4)", font=ctk.CTkFont(size=18, weight="bold"))
        self.header_label.grid(row=0, column=0, pady=(10, 5), padx=20, sticky="w")
        
        # Placeholder for the matplotlib canvas
        self.plot_frame = ctk.CTkFrame(self.main_frame, fg_color="black")
        self.plot_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
        # Big Text Box at the bottom for prediction results
        self.result_frame = ctk.CTkFrame(self.main_frame, height=120)
        self.result_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 20))
        self.result_frame.grid_propagate(False)
        
        self.result_text = ctk.CTkLabel(self.result_frame, text="Result: Click Predict above to decode the mind", font=ctk.CTkFont(size=18, weight="bold"))
        self.result_text.pack(expand=True)

    def load_and_train_subject(self):
        self.status_label.configure(text=f"Status: Training Model for Subj {self.subject}...", text_color="orange")
        self.update()
        
        # 1. Pipeline execution: Load, preprocess & slice
        raw_filtered = load_and_filter_data(subject=self.subject, runs=[3, 4, 7, 8, 11, 12])
        self.epochs = get_epochs(raw_filtered)
        
        # 2. Train classifier specifically calibrated for this subject
        self.model = train_bci_model(self.epochs)
        
        self.status_label.configure(text=f"Status: Model Ready (Subj {self.subject} Active)", text_color="#2ecc71")
        self.current_trial_idx = 0
        self.load_next_trial()

    def on_subject_change(self, selection):
        # Update state based on dropdown selection
        self.subject = int(selection.split(" ")[1])
        self.load_and_train_subject()

    def load_next_trial(self):
        # Reset output text
        self.result_text.configure(text="Result: Waiting for Prediction...", text_color="white")
        
        # Cycle through epochs/trials in the dataset
        if self.epochs:
            self.current_trial_idx = (self.current_trial_idx + 1) % len(self.epochs)
            self.plot_eeg_signals()

    def plot_eeg_signals(self):
        # Clear previous plot if it exists
        for child in self.plot_frame.winfo_children():
            child.destroy()
            
        # Select key motor imagery electrode channels to plot (FC3, Cz, FC4 monitor motor cortex)
        available_channels = self.epochs.ch_names
        channels_to_plot = [ch for ch in ['FC3', 'Cz', 'FC4'] if ch in available_channels]
        if not channels_to_plot:
            channels_to_plot = available_channels[:3]
            
        # Extract data for current trial
        data = self.epochs[self.current_trial_idx].get_data()[0] # Shape: (channels, times)
        times = self.epochs.times
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 3), facecolor='#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        # Plot each target channel's signal
        for ch_name in channels_to_plot:
            idx = available_channels.index(ch_name)
            ax.plot(times, data[idx] * 1e6, label=ch_name, linewidth=1.5) # Scale to microvolts (1e6)
            
        ax.legend(facecolor='#2d2d2d', edgecolor='none', labelcolor='white')
        ax.tick_params(colors='white')
        ax.set_xlabel("Time (Seconds)", color='white')
        ax.set_ylabel("Amplitude (µV)", color='white')
        ax.set_title(f"Trial #{self.current_trial_idx + 1} Raw Waveform", color='white')
        ax.grid(True, color='#444444', linestyle='--')
        
        # Embed the plot inside Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)

    def predict_intent(self):
        # 1. Grab current trial's 2D array: (1, channels, times)
        trial_data = self.epochs[self.current_trial_idx].get_data()
        
        # 2. Get prediction array: [0] or [1]
        prediction = self.model.predict(trial_data)[0]
        
        # Get probability to show confidence
        probabilities = self.model.predict_proba(trial_data)[0]
        confidence = probabilities[prediction] * 100
        
        # 3. Output prediction visually with high contrast indicator
        if prediction == 0:
            self.result_text.configure(
                text=f"👈 PREDICTION: IMAGINING LEFT HAND MOVEMENT\n(Confidence: {confidence:.1f}%)", 
                text_color="#3498db"
            )
        else:
            self.result_text.configure(
                text=f"👉 PREDICTION: IMAGINING RIGHT HAND MOVEMENT\n(Confidence: {confidence:.1f}%)", 
                text_color="#e74c3c"
            )

if __name__ == "__main__":
    app = BCIApp()
    app.mainloop()