# 📡 Wi-Fi Channel State Information (CSI) Sensing 
### *Next-Generation Adaptive Smart Router via Machine Learning*

## 🎯 The Ultimate Goal
We built a machine learning system that uses passive Wi-Fi Channel State Information (CSI) to understand human presence and crowd size in a room. However, instead of stopping at a prediction, we applied this logic to a simulated **Adaptive Wi-Fi Manager**—a smart router script that dynamically cascades its processing, adjusting Transmission (Tx) power and MIMO configurations based on real-time occupancy to save energy and optimize bandwidth.

---

## 🏗️ The "Two-Stage Cascade" Architecture
To balance high accuracy with edge-compute power constraints (Green IoT), our router uses a dual-pipeline architecture:

1. **Stage 1: The "Always-On" Macro Observer (5 GHz IEEE 802.11ax)**
   * **Goal:** Robust Binary Occupancy (Empty vs. Occupied).
   * **How it works:** The router extracts core statistical features from standard 5 GHz Wi-Fi temporal fading. It uses a heavily regularized LightGBM model to detect presence.
   * **Result:** **96% - 99% Leak-Proof Accuracy**. When empty, the router drops to a 10% Tx power "Sleep Mode."

2. **Stage 2: The "Sniper" Micro-Refiner (5 GHz + 60 GHz mmWave Fusion)**
   * **Goal:** High-precision bandwidth allocation (Distinguishing exactly 1 vs. 2 users).
   * **How it works:** If Stage 1 detects occupancy, it temporarily wakes the 60 GHz mmWave phased-array antenna. We use a **Split-Pipeline Sensor Fusion** architecture that applies PCA to compress the noisy 5 GHz subcarriers while preserving the exact spatial Line-of-Sight blockages from the 60 GHz Angle-of-Arrival (AoA) data.
   * **Result:** **93.33% Accuracy** with a 76% reduction in computational feature load. The router perfectly switches between *Focused Beamforming* (1 person) and *Spatial Multiplexing* (2+ people).

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

---
*Developed for the 5G Innovation Hackathon 2026.*
