# 📡 Wi-Fi Channel State Information (CSI) Sensing 
### *Next-Generation Adaptive Smart Router via Machine Learning*

## 🎯 The Ultimate Goal
We built a machine learning system that uses passive Wi-Fi Channel State Information (CSI) to understand human presence and crowd size in a room. However, instead of stopping at a prediction, we applied this logic to a simulated **Adaptive Wi-Fi Manager**—a smart router script that dynamically cascades its processing, adjusting Transmission (Tx) power and MIMO configurations based on real-time occupancy to save energy and optimize bandwidth.

---

## 🏗️ The "Three-Tiered" Smart Router Architecture
To balance high accuracy with edge-compute power constraints (Green IoT) across different environment sizes, our router uses a 3-tiered scalable architecture:

1. **Tier 1: The "Baseline Observer" Profile (Binary Occupancy • IEEE 802.11ax)**
   * **Goal:** Immediate Energy Savings (Green IoT).
   * **How it works:** The router extracts core statistical features from standard 5 GHz Wi-Fi temporal fading to detect basic presence.
   * **Result:** **96% - 99% Leak-Proof Accuracy**. When empty, the router drops to a 10% Tx power "Sleep Mode."

2. **Tier 2: The "Micro-Precision" Profile (High-Precision 1 vs 2 • UC3M 5GHz + 60GHz Fusion)**
   * **Goal:** Precision Bandwidth Allocation for small spaces (Homes/Meeting rooms).
   * **How it works:** If Tier 1 detects occupancy, it temporarily wakes the 60 GHz mmWave antenna. We use a **Split-Pipeline Sensor Fusion** architecture that applies PCA to compress noisy 5 GHz subcarriers while preserving exact spatial Line-of-Sight blockages from 60 GHz Angle-of-Arrival (AoA) data.
   * **Result:** **93.33% Accuracy**. The router perfectly switches between *Focused Beamforming* (1 person) and *Spatial Multiplexing* (2+ people).

3. **Tier 3: The "Capacity Scaling" Profile (Crowd Saturation 0-7 • Device-Free Amplitude)**
   * **Goal:** Heavy Load Balancing for large spaces (Cafes, Classrooms, Open Offices).
   * **How it works:** As rooms fill up, tracing individual bodies becomes physically impossible due to chaotic signal bouncing. The router gracefully adapts into a "Capacity Scaling" state, using a highly robust XGBoost model on raw amplitude bounds to map 0-7 people into 4 **RF Saturation Tiers** (Empty, Low, Medium, Crowd).
   * **Result:** **85-90% Testing Accuracy** across varied room sizes (Tested across Rooms A, B, and C). This dynamically triggers broad network adjustments, like activating secondary Mesh Nodes or throttling guest bandwidth.

---

## 📁 Repository Structure
To evaluate the codebase effectively, please review our notebooks in this exact order. Each notebook corresponds directly to one of our three Smart Router architecture tiers:

*   📓 **`Notebooks/Tier1_Baseline_Observer_IEEE.ipynb`**: Parses raw CSI matrices to simulate our ultra-fast, low-power Binary Occupancy trigger model.
*   📓 **`Notebooks/Tier2_Micro_Precision_UC3M.ipynb`**: Contains the full logic mapping the failure of naive PCA and the success of our custom Split-Pipeline Sensor Fusion model.
*   📓 **`Notebooks/Tier3_Capacity_Scaling_DeviceFree.ipynb`**: Executes the heavy-load XGBoost balancing algorithm, mapping huge feature matrices to actionable RF Saturation abstract classes.
*   📁 **`Notebooks/Failed_and_Legacy_Experiments/`**: We preserved our historical and broken models here to document our engineering learning curve.
*   📦 **`deployed_models/`**: Natively exported models (`.pkl`) ready for embedded deployment.

---

## 🎢 The Journey, The Traps, and The Engineering Solutions

Building a physically accurate RF machine learning model meant falling into—and engineering our way out of—massive technical traps:

### ❌ Trap 1: The "Shiny New Dataset" Illusion
* **The Trap:** We found a glorious IEEE dataset with 16,000+ packets per file, perfect for Crowd Counting!
* **The Reality Check:** Upon reading the abstract, we discovered the dataset only contained scenarios of *Empty Room* vs. *1 Person*. 
* **The Solution:** We split the project. We used the IEEE dataset strictly to perfect the temporal physics of Binary Occupancy and pivoted to the UC3M Multiband dataset to tackle Multi-Class Crowd Counting.

### ❌ Trap 2: Hyperparameter Data Leakage
* **The Trap:** We decreased our sliding window step-size to 10 to generate 3x more training data. Accuracy shot up to 100%. 
* **The Reality Check:** We introduced extreme multicollinearity (90% overlap between windows). Furthermore, our Hyperparameter Tuner was looking at the entire dataset, effectively memorizing the background hardware noise. 
* **The Solution:** We engineered a rigorous nested cross-validation pipeline (`StratifiedGroupKFold`), locking the test files completely out of sight. We proved our final results were physically genuine by running **Permutation Tests** (shuffling labels), which caused the accuracy to rightfully collapse to ~50%.

### ❌ Trap 3: The Curse of Dimensionality (PCA Physics)
* **The Trap:** Fusing 5 GHz and 60 GHz data resulted in 569 features. We threw all 569 into PCA to compress them, but accuracy plummeted. 
* **The Reality Check:** PCA looks for *variance*. 5 GHz bounces everywhere (high variance), but 60 GHz mmWave acts as a precise tripwire (low variance but high importance). PCA treated the critical 60 GHz blockages as "noise" and deleted them.
* **The Solution:** The **Split-Pipeline**. We explicitly sliced the matrices, applying PCA *only* to the 500 subcarriers of the 5 GHz band (compressing them to ~63 features) and horizontally concatenated them with the raw, untouched 60 GHz AoA data. Accuracy surged to **93.33%** while maintaining edge-compute efficiency.

### ❌ Trap 4: The Complexity & PCA vs. Decision Tree Clash
* **The Trap:** To count massive crowds (0-7 people on a device-free dataset), we initially used PCA to reduce 360 raw amplitude features. Later, we attempted injecting advanced signal physics (Level Crossing Rate, Kurtosis) to create a massive 720-feature super-matrix, expecting performance to skyrocket.
* **The Reality Check:** PCA mathematically mixes physical subcarriers into diagonal combinations, completely ruining XGBoost's ability to create discrete, axis-aligned spatial splits. And the advanced "distribution shape" features? They merely amplified hardware micro-noise instead of mapping human bodies.
* **The Solution:** We discarded PCA entirely and aggressively pruned our feature engineering back to the simplest, most robust physical bounding box: **Raw Amplitude Variance (Mean, Std, Max, Min)** across 360 strictly native dimensions. By relying on XGBoost's internal `colsample_bytree` to hunt pristine subcarriers, and mapping the 0-7 occupancy into 4 physical **"RF Saturation Tiers"** (Empty, Low, Medium, Crowd), our out-of-the-box accuracy confidently anchored at **85-90% across varied room sizes**.

---

## 🛠️ The Tech Stack
* **Data Processing:** `numpy`, `pandas`, `scipy.io` (MATLAB extraction)
* **Machine Learning:** `LightGBM`, `XGBoost`, `scikit-learn` (PCA, StandardScaler, GroupKFold)
* **Visualization:** `matplotlib`, `seaborn`

## 📊 Key Results Summary
| Task | Dataset | Algorithm | Best Accuracy | Key Feature |
| :--- | :--- | :--- | :--- | :--- |
| **Binary Occupancy** | IEEE 802.11ax (Temporal) | Tuned LightGBM | **99.14%** | Window Step-Size Optimization |
| **General Crowd Sizing (0-4)** | UC3M 5 GHz (Spatial) | LightGBM + PCA | **91.02%** | Spatial PCA Fingerprinting |
| **High-Precision Counting (1 vs 2)** | UC3M 5GHz + 60GHz Fusion | Split-Pipeline LGBM | **93.33%** | mmWave AoA Blockage Fusion |
| **High-Density Crowd Saturation (0-7)** | Device-Free Wi-Fi (Amplitude) | Tuned XGBoost (No PCA) | **85-90%** | Raw Physical Bounding (Mean/Std/Max/Min) |

---
*Developed for the 5G Innovation Hackathon 2026.*
