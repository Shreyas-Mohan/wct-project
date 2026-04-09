import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

class RoomCalibrator:
    """
    Handles Environment Dependency for Wi-Fi Passive Sensing.
    Removes Static Clutter (walls, furniture), fits dynamic room variance scaling,
    and performs Few-Shot Model Adaptation for adaptive energy savings.
    """
    def __init__(self, global_model=None):
        self.baseline_h_static = None
        self.scaler = StandardScaler()
        
        # If a pre-trained global model is present, we could do Neural Network fine-tuning.
        # For this template, if none is provided, we use a lightweight ML classifier 
        # (RandomForest) to demonstrate fast edge-training on the new room data.
        self.model = global_model if global_model else RandomForestClassifier(n_estimators=50, random_state=42)
        self.is_calibrated = False

    def calibrate_empty_room(self, empty_csi_data):
        """
        Phase 1: Capture the Empty Room Baseline (static clutter).
        Calculates the mean signal reflection of the empty room to serve as the H_static matrix.
        """
        print("Calibrating Empty Room (0 occupants)...")
        # Average across the temporal axis to get a 1D vector of static feature baselines
        self.baseline_h_static = np.mean(empty_csi_data, axis=0)
        return self.baseline_h_static

    def remove_static_clutter(self, csi_data):
        """
        Subtract the static baseline fromlive CSI data.
        H_dynamic = H_new - H_static
        """
        if self.baseline_h_static is None:
            raise ValueError("System not calibrated for empty room yet! Please run Phase 1.")
        
        # Mathematically "deletes" the room's static architecture, isolating movement
        return csi_data - self.baseline_h_static

    def fit_environment_scaler(self, dynamic_data):
        """
        Phase 2: Fit scaler bounds on the dynamic variance of the specific room.
        Signals drop differently based on router distance. We need local Min/Max boundaries.
        """
        print("Fitting scaler to room's unique dynamic RF variance...")
        self.scaler.fit(dynamic_data)

    def run_full_calibration_wizard(self, empty_data, occupied_1_data, occupied_2_data=None):
        """
        Main setup function taking the Arrays collected during the 3-minute Wizard.
        empty_data: CSI matrix when room is 0 people.
        occupied_1_data: CSI matrix when user walks heavily around room for 1 min.
        occupied_2_data: (Optional) CSI matrix when 2 people are in the room.
        """
        print("Starting Setup Wizard Processing...")
        
        # 1. Establish Empty Baseline
        self.calibrate_empty_room(empty_data)
        
        # 2. Remove static clutter from all collected training samples
        empty_dyn = self.remove_static_clutter(empty_data)
        occ_1_dyn = self.remove_static_clutter(occupied_1_data)
        
        # Combine data and create Ground Truth labels
        X_train = np.vstack((empty_dyn, occ_1_dyn))
        y_train = np.concatenate((np.zeros(empty_data.shape[0]), np.ones(occupied_1_data.shape[0])))
        
        if occupied_2_data is not None:
            occ_2_dyn = self.remove_static_clutter(occupied_2_data)
            X_train = np.vstack((X_train, occ_2_dyn))
            y_train = np.concatenate((y_train, np.full(occupied_2_data.shape[0], 2)))

        # 3. Fit Local Scaling
        self.fit_environment_scaler(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        # 4. Train / Few-Shot Adapt model
        print("Adapting model to the new room environment via Transfer Learning / Fitting...")
        self.model.fit(X_train_scaled, y_train)
        
        self.is_calibrated = True
        print("✅ Calibration Complete. Adaptive Energy Saving is now Active for this specific room.")

    def predict_occupancy(self, live_data):
        """
        Process incoming live stream data and output occupancy state.
        Ensures the data perfectly aligns with the calibrated room boundaries before predicting.
        """
        if not self.is_calibrated:
            raise ValueError("System must be fully calibrated before predicting!")
        
        # 1. Remove Room's walls/furniture
        live_dyn = self.remove_static_clutter(live_data)
        
        # 2. Scale using local boundaries
        live_scaled = self.scaler.transform(live_dyn)
        
        # 3. Predict
        return self.model.predict(live_scaled)
        
    def auto_update_baseline(self, sustained_night_data, learning_rate=0.05):
        """
        Phase 3: Continuous Background Adaptation.
        If the room detects zero motion variance for a prolonged time (e.g., 3 AM),
        gently update the expected baseline to account for moved chairs or new boxes.
        """
        if self.baseline_h_static is None:
            return
            
        current_empty_mean = np.mean(sustained_night_data, axis=0)
        
        # Exponential Moving Average for Silent Self-Healing
        self.baseline_h_static = (1 - learning_rate) * self.baseline_h_static + (learning_rate * current_empty_mean)
        print("Updated structural baseline (Silent Self-Healing applied) - compensating for minor physical room shifts.")

    def save_room_profile(self, filename="room_rf_profile.pkl"):
        """Save the room's RF profile so calibration persists across system reboots."""
        if not self.is_calibrated:
            print("Nothing to save.")
            return
        
        data_to_store = {
            'baseline': self.baseline_h_static,
            'scaler': self.scaler,
            'model': self.model
        }
        joblib.dump(data_to_store, filename)
        print(f"Room profile securely saved to '{filename}'.")

    def load_room_profile(self, filename="room_rf_profile.pkl"):
        data = joblib.load(filename)
        self.baseline_h_static = data['baseline']
        self.scaler = data['scaler']
        self.model = data['model']
        self.is_calibrated = True
        print(f"Room profile successfully loaded from '{filename}'. Ready to predict!")

# ==========================================
# EXAMPLE USAGE / SIMULATION
# ==========================================
if __name__ == "__main__":
    print("--- Simulating the Calibration Wizard ---")
    
    # Simulating data matrices from Sensor Extraction (Temporal Rows x Features)
    # We pretend our feature space has 569 columns (from earlier 5G+60G fusion)
    mock_features = 569
    
    # 1. User leaves room, presses "Calibrate": 60 seconds (eg. 100 measurements)
    mock_empty_data = np.random.normal(loc=0.5, scale=0.01, size=(100, mock_features))
    
    # 2. User walks around individually: 60 seconds
    mock_1person_data = np.random.normal(loc=0.8, scale=0.1, size=(100, mock_features))
    
    # Initialize the engine
    calibrator = RoomCalibrator()
    
    # Run setup
    calibrator.run_full_calibration_wizard(mock_empty_data, mock_1person_data)
    
    print("\n--- Simulating Live Runtime ---")
    # Live data streams in (eg. 5 measurements)
    live_stream = np.random.normal(loc=0.8, scale=0.1, size=(5, mock_features))
    occupancy_predictions = calibrator.predict_occupancy(live_stream)
    
    print(f"Live Predictions: {occupancy_predictions}")
    
    # System performs nighttime maintenance
    mock_night_data = np.random.normal(loc=0.51, scale=0.01, size=(200, mock_features))
    calibrator.auto_update_baseline(mock_night_data)
    
    calibrator.save_room_profile()