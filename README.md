# Banking Classifier

## Overview
This project delivers:
1) A classifier to predict whether income is > $50K  
2) A marketing segmentation model based on demographics and income potential

---

## Required Files
Make sure these files are in the same project folder:

- `census-bureau.data`
- `census-bureau.columns`
---

## Environment Setup
Recommended (conda):
1. Create environment  
   `conda create -n jpm_project python=3.13 -y`
2. Activate  
   `conda activate jpm_project`
---

## How to Run
1. Launch Jupyter:
   - `jupyter notebook`

2. Open `JPM_project.ipynb`

3. Run cells **in order**, top to bottom.

---

## Notebook Execution Flow (Detailed)

### 1. Data Loading
- Loads column names from `census-bureau.columns`
- Loads data from `census-bureau.data`
- Shows a preview (`df.head()`)

### 2. Sanity Checks
- Verifies all rows have 42 fields
- Reports dtype counts
- Reports missing values
- Shows label distribution

### 3. Preprocessing & Train/Validation Split
- Creates binary target from `label`
- Separates `weight` as `sample_weight`
- Defines numeric vs categorical features
- Builds preprocessing pipeline (imputation + encoding)
- Groups rare categories via `min_frequency` to reduce sparsity
- Splits into train/validation (stratified)

### 4. Fast Baseline Model (SGD Logistic Regression)
- Tunes regularization strength (`alpha`)
- Evaluates ROC‑AUC, PR‑AUC, Precision, Recall, F1, Accuracy
- Optimizes threshold using F1

### 5. Tree‑Based Model Comparison
- Trains RandomForest and HistGradientBoosting (HGB)
- Compares metrics to baseline
- Selects HGB as final classifier

### 6. Final HGB + Threshold Tuning
- Final HGB model training
- Best threshold selection (F1‑max)
- Final evaluation metrics and confusion matrix

### 6A. Probability Calibration
- Calibrates probabilities (Brier score improvement)
- Produces business‑interpretable income probabilities

### 6B. ROI‑Based Threshold Selection
- Chooses threshold based on campaign economics
- Reports expected value and precision/recall at ROI‑optimal threshold

### 6C. Weighted Cross‑Validation
- 3‑fold weighted ROC‑AUC / PR‑AUC for stability

### 6D. Feature Importance (Permutation)
- Estimates top drivers of model performance

### 7. Segmentation Dataset Preparation
- Adds `income_prob` (HGB calibrated probability)
- Builds clustering feature set

### 8. Preprocess + Dimensionality Reduction
- One‑hot encoding
- TruncatedSVD embedding

### 9. Cluster Selection and Training
- Selects k via silhouette/inertia
- Final KMeans clustering

### 9A. Cluster Stability
- ARI stability across seeds

### 10. Cluster Profiling
- Weighted summaries with medians/log transforms
- Top categories by cluster

### 10A. Persona Naming
- Business‑friendly segment names

### 11. Operationalization Notes
- Scoring, threshold policy, drift monitoring, refresh cadence

---

## Outputs (from notebook)
- Classification metrics: ROC‑AUC, PR‑AUC, Precision, Recall, F1, Accuracy  
- Final model selection: **HistGradientBoosting**
- Calibrated probabilities and ROI‑optimal threshold
- Cluster assignments + cluster profiles for segmentation

