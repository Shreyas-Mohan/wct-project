import time
import numpy as np
import datetime
from room_calibration import RoomCalibrator

def daemon_start():
    print("================================================================")
    print("🌙 Nighttime Background Self-Healing Daemon Active 🌙")
    print("Running silently on edge router to check for physical layout shifts")
    print("================================================================")

    # Initialize the Calibrator (Assume the room already went through ARC)
    try:
        calibrator = RoomCalibrator(global_model_path="deployed_models/tier2_micro_precision.pkl")
        calibrator.load_room_profile("deployed_models/my_living_room_profile.pkl")
        print("\n[DAEMON] Room Profile Loaded successfully.")
    except Exception as e:
        print(f"\n[DAEMON ERROR] {e}. \nMake sure to run setup_wizard_cli.py first.")
        return

    # Loop forever measuring standard variance
    print("\n[DAEMON] Monitoring live variance (Checking for prolonged flatlines)...")
    
    mock_features = 569
    
    # 5 iterations of simulated night-time polling
    for hour in range(1, 6):
        time.sleep(1.5)  # Fast-forward simulation of 1 hour passes
        
        # Simulate Nighttime flatline CSI Data (Slightly shifted from setup day due to a moved chair)
        # Setup was loc=0.5. The moved chair makes new empty baseline = 0.52
        simulated_hourly_block = np.random.normal(loc=0.52, scale=0.015, size=(100, mock_features))
        
        # Compute Variance 
        variance = np.var(simulated_hourly_block)
        
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Hourly Check #{hour} - Signal Variance: {variance:.5f}")
        
        if variance < 0.0005: 
            print("   -> 🛏️ Zero motion detected for sustained period (Room is empty).")
            # Trigger Exponential Moving Average
            calibrator.auto_update_baseline(simulated_hourly_block, learning_rate=0.05)
        else:
            print("   -> 🚶 Motion detected. Ignoring Self-Healing.")

    # Save memory to flash
    print("\n[DAEMON] Saving adjusted structural boundaries back to flash memory...")
    calibrator.save_room_profile("my_living_room_profile.pkl")
    print("\n✅ Nightly self-healing complete. System resilient to furniture shifts.")

if __name__ == "__main__":
    daemon_start()
