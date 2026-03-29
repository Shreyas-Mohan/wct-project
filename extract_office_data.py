import os
import shutil

# --- Configuration ---
source_base = "20181211"
destination_folder = "Widar-dataset-office"
target_users = ["user7", "user8", "user9"]

# 1 = Push & Pull, 2 = Sweep, 4 = Slide
target_gestures = ["1", "2", "4"] 
target_receiver = "r1"

# The file flagged as corrupted in the Widar 3.0 Bug Notice
corrupted_files = ["user9-1-1-1-1-r1.dat"]

# --- Execution ---
# Create destination folder if it doesn't already exist
os.makedirs(destination_folder, exist_ok=True)

copied_count = 0

print(f"Scanning '{source_base}' for target files...")

for user in target_users:
    user_folder = os.path.join(source_base, user)
    
    # Check if the user folder exists
    if not os.path.exists(user_folder):
        print(f"  -> Warning: Folder '{user_folder}' not found. Skipping.")
        continue
        
    for filename in os.listdir(user_folder):
        if not filename.endswith(".dat"):
            continue
            
        # File naming format: id-a-b-c-d-Rx.dat
        # parts[1] is the gesture, parts[5] is the receiver
        parts = filename.replace(".dat", "").split("-")
        
        # Safety check to avoid index errors on weirdly named files
        if len(parts) < 6:
            continue
            
        gesture = parts[1]
        receiver = parts[5]
        
        # Apply our exact project filters
        if gesture in target_gestures and receiver == target_receiver:
            
            # Block the known corrupted file
            if filename in corrupted_files:
                print(f"  -> Skipping known corrupted file: {filename}")
                continue
                
            source_path = os.path.join(user_folder, filename)
            dest_path = os.path.join(destination_folder, filename)
            
            # Copy the file to the new working directory
            shutil.copy2(source_path, dest_path)
            copied_count += 1

print("-" * 30)
print(f"Extraction Complete!")
print(f"Successfully copied {copied_count} files into '{destination_folder}'.")