# 📡 Wi-Fi Channel State Information (CSI) Sensing Project

## 🎯 The Ultimate Goal
We are building a machine learning system that uses Wi-Fi Channel State Information (CSI) to understand human presence in a room. But instead of stopping at just "predicting an output," we are using those predictions to simulate an **Adaptive Wi-Fi Manager**—a smart router script that dynamically adjusts its Transmission (Tx) power and MIMO configurations based on how many people are in the room to save energy and optimize bandwidth.

---

## 🎢 The Journey, The Traps, and The Misconceptions

If you're reading this, just know we did not get here linearly. We fell into a few massive rabbit holes:

### ❌ Trap 1: The "Temporal Fading" Error
We initially started with the **Multiband Wi-Fi Passive Sensing Dataset** and tried to extract temporal-fading metrics (moving variance over windows of 100 packets). 
* **The Reality Check:** The code crashed repeatedly because the dataset files only contained 50 packets each! 
* **The Lesson:** We confused *Temporal Sensing* (capturing the Doppler shift of a moving human over time) with *Static Spatial Sensing* (capturing the distorted physical "shape" of the Wi-Fi frequencies bouncing off static bodies). Once we shifted our math to extract the raw spatial shape of the 500 subcarriers instead of time-based variance, the whole pipeline unlocked.

### ❌ Trap 2: The "Shiny New Dataset" Illusion
Later on, we found a glorious **IEEE 802.11ax (Wi-Fi 6)** dataset. It had 996 subcarriers, 16,000+ packets per file, and actual tracking of people walking and running! Perfect for Crowd Counting, right?
* **The Reality Check:** I read the abstract. The dataset only contained scenarios of *Empty Room* vs. *1 Person* doing activities. 
* **The Lesson:** You cannot train an ML model to count 0 to 4 people if it has never seen 2, 3, or 4 people. Machine learning cannot magically extrapolate multipath wave intersections it hasn't seen.

---

## 🛠️ What We've Actually Built So Far (The Beta)

While working on the static 5 GHz Multiband dataset (0 to 4 people), we built a fully functioning pipeline:

1. **Parser:** Engineered a custom data loader that converts chaotic MATLAB complex strings (`0.5+0.2i`) into pure Python float amplitudes.
2. **Feature Engineering:** Extracted all 500 spatial subcarriers + **8 statistical meta-features** (Mean, Min, Max, Peak-to-Peak, Skewness, Kurtosis, etc.).
3. **Dimensionality Reduction:** Pushed 500 subcarriers through **PCA** to keep 95% of the variance resulting in 94 core, un-correlated components.
4. **Balancing:** Used **SMOTE** to fix the highly imbalanced dataset (which followed a binomial distribution naturally).
5. **The Model:** Ran a Randomized Grid Search on `LightGBM` achieving a whopping **91.02% Accuracy** on multi-class prediction.
6. **The Application:** Wrote an `Adaptive Wi-Fi Manager Simulation` that takes the model's live outputs and logs exact networking responses (e.g., *Status: Empty -> Reducing Tx Power to 20%; Status: 4 People -> Max Tx, Activating MU-MIMO*).

---

## 🚀 The Final "Two-Part" Plan

To align with high-level networking research papers while still implementing our cool Smart Router idea, we decided to split the project into two distinct parts:

### Part 1: Robust Binary Occupancy Detection (The Core System)
**Goal:** Prove high-fidelity room occupancy using state-of-the-art temporal CSI data.
* **Dataset:** The IEEE 802.11ax (Wi-Fi 6) dataset constraint.
* **Focus:** Extreme efficiency. Using tree-based models (LightGBM/XGBoost) to detect Human (Walking/Running) vs. Empty Space with near-perfect accuracy using temporal time-series features.
* **Status:** *Waiting on IEEE DataPort access.* ⏳

### Part 2: Adaptive Router & Multi-Person Capacity (The Beta Feature)
**Goal:** Show the real-world application layer of this math. 
* **Dataset:** The static 5 GHz Multiband dataset (0-4 people).
* **Focus:** Because robust temporal datasets with multiple people don't widely exist yet, this serves as a mathematical proxy/beta. We use our 91% accurate 5-class LightGBM model to trigger a simulated IoT network controller. 
* **Status:** *100% Complete & Ready for Presentation.* ✅
