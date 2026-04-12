import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import time
import os

class RoomCalibrator:
    """
    Active Room Calibration (ARC) Protocol for Wi-Fi Passive Sensing.
    Removes Static Clutter (walls, furniture) via ΔCSI Math, fits dynamic room variance scaling,
    and performs Few-Shot Model Adaptation (Transfer Learning) for edge deployment.
    """
    def __init__(self, global_model_path=None):
        self.baseline_h_static = None
        self.scaler = StandardScaler()
        
        # Load the pre-trained Global Base Model (Tier 2 Split-Pipeline) if available
        if global_model_path and os.path.exists(global_model_path):
            print(f"[ARC] Loading Global Base Model from {global_model_path}...")
            # Unpack the tuple (model, pca_5g, scaler_5g, scaler_60g) saved earlier
            self.global_artifacts = joblib.load(global_model_path)
            self.base_model = self.global_artifacts[0]
        else:
            print("[ARC] No Base Model found. Initializing new Lightweight Edge Classifier...")
            self.base_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, verbosity=-1)
            self.global_artifacts = None
            
        self.adapted_model = None
        self.is_calibrated = False

    def calibrate_empty_room(self, empty_csi_data):
        """
        Phase 1: Capture the Empty Room Baseline (static clutter).
        Calculates the mean signal reflection of the empty room to serve as the H_static matrix.
        """
        print("[ARC] Calibrating Empty Room (0 occupants)...")
        # Average across the temporal axis to get a 1D vector of static feature baselines
        self.baseline_h_static = np.mean(empty_csi_data, axis=0)
        return self.baseline_h_static

    def remove_static_clutter(self, csi_data):
        """
        Phase 2: Subtract the static baseline from live CSI data (ΔCSI Math).
        H_dynamic = H_new - H_static
        """
        if self.baseline_h_static is None:
            raise ValueError("System not calibrated for empty room yet! Please run Phase 1.")
        
        # Mathematically "deletes" the room's static architecture, isolating only human movement
        return csi_data - self.baseline_h_static

    def fit_environment_scaler(self, dynamic_data):
        """
        Phase 3: Fit scaler bounds on the dynamic variance of the specific room.
        Signals drop differently based on router distance. We need local Min/Max boundaries.
        """
        print("[ARC] Fitting scaler to room's unique dynamic RF variance...")
        self.scaler.fit(dynamic_data)

    def run_full_calibration_wizard(self, empty_data, occupied_1_data, occupied_2_data=None):
        """
        Main setup function executed by the user during the initial 2-minute Wi-Fi installation.
        """
        print("\n--- Starting Active Room Calibration (ARC) Wizard ---")
        
        # 1. Establish Empty Baseline
        self.calibrate_empty_room(empty_data)
        
        # 2. Subtract Static Clutter (ΔCSI)
        empty_dyn = self.remove_static_clutter(empty_data)
        occ_1_dyn = self.remove_static_clutter(occupied_1_data)
        
        # Construct Training Set
        X_train = np.vstack((empty_dyn, occ_1_dyn))
        y_train = np.concatenate((np.zeros(empty_data.shape[0]), np.ones(occupied_1_data.shape[0])))
        
        if occupied_2_data is not None:
            occ_2_dyn = self.remove_static_clutter(occupied_2_data)
            X_train = np.vstack((X_train, occ_2_dyn))
            y_train = np.concatenate((y_train, np.full(occupied_2_data.shape[0], 2)))

        # 3. Fit Local Scaling boundaries
        self.fit_environment_scaler(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        # 4. Few-Shot Transfer Learning (Adaptation)
        print("[ARC] Adapting Global Model to the new room environment via Transfer Learning...")
        try:
            # If we have a real LGBM booster, we can use it as init_model
            booster = self.base_model.booster_
            self.adapted_model = lgb.LGBMClassifier(n_estimators=20, learning_rate=0.01, max_depth=3, verbosity=-1)
            self.adapted_model.fit(X_train_scaled, y_train, init_model=booster)
        except AttributeError:
            # Fallback if no pre-trained model was loaded
            self.adapted_model = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=5, verbosity=-1)
            self.adapted_model.fit(X_train_scaled, y_train)
        
        self.is_calibrated = True
        print("✅ [ARC] Calibration Complete. Model is deeply customized to this specific room's topography.")

    def predict_occupancy(self, live_data):
        """
        Inference logic for Edge Router runtime.
        """
        if not self.is_calibrated:
            raise ValueError("System must be fully calibrated before predicting!")
        
        # 1. Remove Room's fixed objects/walls (ΔCSI)
        live_dyn = self.remove_static_clutter(live_data)
        
        # 2. Scale
        live_scaled = self.scaler.transform(live_dyn)
        
        # 3. Predict using the newly localized, few-shot adapter
        return self.adapted_model.predict(live_scaled)
        
    def auto_update_baseline(self, sustained_night_data, learning_rate=0.05):
        """
        Background Self-Healing Daemon logic.
        Runs automatically at 3 AM to quietly compensate for moved furniture or new objects.
        EMA (Exponential Moving Average) prevents drastic, sudden model corruption.
        """
        if self.baseline_h_static is None:
            return
            
        current_empty_mean = np.mean(sustained_night_data, axis=0)
        
        # Exponential Moving Average for Silent Self-Healing
        self.baseline_h_static = (1 - learning_rate) * self.baseline_h_static + (learning_rate * current_empty_mean)
        print(f"[Daemon] Updated structural baseline (Silent Self-Healing applied) - accommodating minor physical room shifts.")

    def save_room_profile(self, filename="room_rf_profile.pkl"):
        if not self.is_calibrated:
            print("[ARC] Nothing to save.")
            return
        
        data_to_store = {
            'baseline': self.baseline_h_static,
            'scaler': self.scaler,
            'adapted_model': self.adapted_model
        }
        os.makedirs('deployed_models', exist_ok=True)
        path = os.path.join('deployed_models', filename)
        joblib.dump(data_to_store, path)
        print(f"[ARC] Local Room Profile safely serialized to '{path}' for persistent reboot.")

# ==========================================
# EXAMPLE USAGE / SIMULATION
# ==========================================
if __name__ == "__main__":
    print("--- Simulating the Active Room Calibration Pipeline ---")
    mock_features = 569 # Fused 5G + 60G vector
    
    # 1. User sets up router, room is empty
    mock_empty_data = np.random.normal(loc=0.5, scale=0.01, size=(50, mock_features))
    
    # 2. User walks around individually to seed the dynamic threshold
    mock_1person_data = np.random.normal(loc=0.8, scale=0.1, size=(50, mock_features))
    
    calibrator = RoomCalibrator(global_model_path="deployed_models/tier2_micro_precision.pkl")
    calibrator.run_full_calibration_wizard(mock_empty_data, mock_1person_data)
    
    print("\n--- Simulating Live Runtime (Edge Router) ---")
    live_stream = np.random.normal(loc=0.8, scale=0.1, size=(3, mock_features))
    start_time = time.time()
    occupancy_predictions = calibrator.predict_occupancy(live_stream)
    print(f"Inference Latency: {(time.time() - start_time)*1000:.3f} ms. Live Predictions: {occupancy_predictions}")
    
    calibrator.save_room_profile()
