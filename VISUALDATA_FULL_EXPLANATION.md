# VisualData Pipeline: Full Explanation (Start to Finish)

## 1) What This Project Is

VisualData is an automated image classification pipeline. It accepts an image dataset, prepares the data, trains multiple CNN architectures, compares them using multiple metrics, and automatically selects the best model. The final output includes:
- Best architecture name
- Performance metrics
- Saved model file (.pkl)

This document explains the entire pipeline from scratch, why each step exists, how it works, and what the user gets at the end.

---

## 2) Big Picture: What The Pipeline Does

1. Load dataset (CIFAR-10 or custom images)
2. Clean and normalize images
3. Balance class distribution using SMOTE (if needed)
4. Convert data into PyTorch tensors
5. Build a list of candidate CNN architectures
6. Use Optuna to tune hyperparameters
7. Train all candidate models
8. Evaluate each model with multiple metrics
9. Select the best model by multi-criteria scoring
10. Save artifacts (best model + metrics)

---

## 3) Inputs

### Supported dataset types
1. CIFAR-10 (built-in)
2. Custom dataset with folder structure:

```
custom_dataset/
├── class_a/
│   ├── img1.jpg
│   └── img2.png
├── class_b/
│   ├── img1.jpg
│   └── img2.png
```

### Key configuration variables
- DATASET_TYPE: "CIFAR10" or "CUSTOM"
- CUSTOM_DATASET_PATH: path to folder (only if CUSTOM)
- TARGET_SIZE: image resize size (32 x 32)
- TEST_SPLIT: test size ratio (0.2)

---

## 4) Step-by-Step Pipeline Explanation

### Step 1: Install Dependencies
**Why:** Training, evaluation, balancing, and visualization require multiple libraries.
**What it does:** Installs PyTorch, scikit-learn, Optuna, SMOTE, etc.

### Step 2: Import Libraries + GPU Detection
**Why:** Detect GPU so training uses CUDA if available.
**What it does:** Loads libraries and chooses device = cuda or cpu.

### Step 3: Load Dataset
**Why:** Pipeline must load images and labels before training.
**What it does:**
- CIFAR-10: downloads and loads automatically
- CUSTOM: reads folders and labels based on directory names

### Step 4: Smart Dataset Analysis (Auto Architecture Selection)
**Why:** Not all architectures fit all dataset sizes.
- Small datasets overfit if models are too large.
- Large datasets need stronger models.

**What it does:**
- Counts number of training samples
- Classifies dataset as SMALL, MEDIUM, or LARGE
- Chooses architecture subset accordingly
- Adjusts epochs, Optuna trials, early stopping patience

### Step 5: Data Cleaning and Normalization
**Why:** Corrupt images cause crashes and noisy training.
**What it does:**
- Filters invalid images
- Converts values to [0,1]
- Ensures consistent shape

### Step 6: Class Balancing (SMOTE)
**Why:** Imbalanced datasets create biased models.
**What it does:**
- Applies SMOTE to generate synthetic samples for minority classes
- Uses smaller k_neighbors for small datasets
- Skips SMOTE if too few samples

### Step 7: Visualization
**Why:** Confirms dataset is loaded correctly.
**What it does:** Displays sample images.

### Step 8: Convert to PyTorch Tensors
**Why:** PyTorch expects channel-first tensors.
**What it does:**
- Converts (H,W,C) to (C,H,W)
- Creates train/test tensors

### Step 9: Define Architectures
**Why:** Different CNNs trade speed vs accuracy.
**What it does:** Builds a list of models such as:
- SimpleCNN
- MediumCNN
- DeepCNN
- CustomResNet
- CustomDenseNet
- MobileNetV2, SqueezeNet, ShuffleNetV2, EfficientNetB0

### Step 10: Hyperparameter Optimization (Optuna)
**Why:** Optimal learning rate and batch size differ by dataset.
**What it does:**
- Searches best learning rate, batch size, weight decay
- Runs short trials per architecture
- Uses early stopping to save time

### Step 11: Training Each Model
**Why:** Compare multiple architectures fairly.
**What it does:**
- Train every selected architecture with tuned hyperparameters
- Track validation metrics

### Step 12: Evaluate Models
**Why:** One metric is not enough.
**What it does:**
- Computes accuracy and F1-score
- Measures latency (inference speed)
- Measures memory usage (model size)
- Measures training time

### Step 13: Multi-Criteria Selection
**Why:** Best model is not always highest accuracy.
**What it does:**
- Normalizes metrics
- Uses weighted score:
  - Accuracy (35%)
  - F1-score (25%)
  - Latency (15%)
  - Memory (15%)
  - Training time (10%)

### Step 14: Save Outputs
**Why:** Model must be reusable in deployment.
**What it does:**
- Saves best model (.pkl)
- Saves metrics JSON
- Saves CSV of all model results

### Step 15: Final Report
**Why:** Clear summary for human users.
**What it does:** Prints dataset details, best model, and key metrics.

---

## 5) Outputs (What User Gets)

1. Best model file
   - visualdata_output/<best_architecture>_best_model.pkl

2. Best metrics JSON
   - visualdata_output/best_model_metrics.json

3. Full results CSV
   - visualdata_output/all_results.csv

---

## 6) Why This Design Works

- **Automation:** User does not manually choose architecture.
- **Fair comparison:** Same training and evaluation pipeline for all models.
- **Balanced scoring:** Best model considers performance + efficiency.
- **Deployment ready:** Outputs are saved for inference immediately.

---

## 7) How Users Interact With The System

### Option A: Notebook workflow (training)
- User runs the pipeline in Colab
- Pipeline produces model artifacts

### Option B: API workflow (user-driven training)
- User uploads dataset to FastAPI
- Pipeline trains in backend
- User downloads best model + metrics

---

## 8) Key Takeaways For Presentation

- VisualData is a full ML lifecycle pipeline: data -> training -> evaluation -> deployment.
- Smart architecture selection reduces unnecessary training time.
- Multi-criteria decision avoids picking only the most accurate but slow model.
- Outputs are production-ready for inference and API integration.

---

## 9) Short Summary You Can Say To Your Professor

"VisualData automatically trains multiple CNN architectures on an input dataset, tunes them with Optuna, evaluates them across accuracy, F1, latency, memory, and training time, and then selects the best model using a weighted scoring system. The final result is a saved model file and metrics that can be deployed through an API for real users."
