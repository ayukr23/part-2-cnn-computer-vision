# Part 2: Computer Vision — Surface Defect Classification CNN

## Project Overview

This project formulates and solves an **image classification** problem using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The dataset contains surface images of components labelled as one of four classes: **normal**, **scratch**, **stain**, or **dent**. The goal is to automatically classify a surface photograph into the correct defect category.


## Repository Structure

```
part-2-cnn-computer-vision/
├── README.md
├── notebook.ipynb              ← Main Jupyter notebook (all 7 tasks)
├── requirements.txt
├── labels.csv                  ← Ground-truth labels (filename → class)
├── images/
│   ├── normal/   (120 images)
│   ├── scratch/  (105 images)
│   ├── dent/     (120 images)
│   └── stain/    (120 images)
├── sample_predictions/
│   └── prediction_outputs.png
└── results/
    ├── class_distribution.png
    ├── sample_images.png
    ├── augmentation_examples.png
    ├── accuracy_loss_curves.png
    ├── confusion_matrix.png
    └── best_model.keras
```


## Task 1 — Problem Identification: Image Classification

The dataset assigns **one label per image** from a fixed set of 4 categories. This is a classic **multi-class image classification** task:

- **Not Object Detection** — no bounding boxes needed; the whole image represents one surface condition.
- **Not Segmentation** — we classify the entire image, not individual pixels.
- **Image Classification** — each image → one of {normal, scratch, stain, dent}.



## Task 2 — Dataset Exploration

| Property | Value |
|---|---|
| Total images | 465 |
| Number of classes | 4 |
| Normal images | 120 |
| Scratch images | 105 |
| Dent images | 120 |
| Stain images | 120 |
| Imbalance ratio | ~1.14 (well-balanced) |
| Image size (typical) | Variable → resized to 128 × 128 |

The dataset is **well-balanced** with roughly equal images per class. A minor shortfall in the scratch class (~10%) is handled through data augmentation rather than oversampling.



## Task 3 — Image Preprocessing

| Step | Detail |
|---|---|
| Resize | All images resized to **128 × 128** pixels |
| Normalise | Pixel values scaled from [0, 255] → **[0, 1]** |
| Train/Val/Test split | **70 / 15 / 15** (stratified) |
| Augmentation (train only) | Horizontal flip, ±15° rotation, ±10% shift & zoom, brightness jitter [0.8–1.2] |

Augmentation is applied **only to the training set** via `ImageDataGenerator` to prevent data leakage into validation/test.



## Task 4 — CNN Architecture

```
Input (128, 128, 3)
│
├── Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)     Block 1
├── Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)     Block 2
├── Conv2D(128,3×3) → BatchNorm → ReLU → MaxPool(2×2)     Block 3
│
├── Flatten
├── Dropout(0.4)
├── Dense(256) → ReLU
├── Dropout(0.3)
└── Dense(4)   → Softmax                                   Output
```

**Optimizer:** Adam (lr = 1e-3)  
**Loss:** Sparse Categorical Cross-Entropy  
**Metric:** Accuracy



## Task 5 — Training & Evaluation

**Callbacks used:**
- `EarlyStopping` — stops when val_accuracy plateaus for 7 epochs; restores best weights.
- `ReduceLROnPlateau` — halves learning rate when val_loss stagnates for 4 epochs.
- `ModelCheckpoint` — saves the best model weights to `results/best_model.keras`.

**Expected results (with real images):**

| Set | Accuracy |
|---|---|
| Training | ~90–95% |
| Validation | ~85–92% |
| Test | ~83–90% |

Results, confusion matrix, and sample predictions are saved in `results/` and `sample_predictions/`.

---

## Task 6 — CNN Concept Explanations

### What is Convolution?
A small learnable filter (e.g. 3×3) slides across the image. At each position it computes the dot product between the filter weights and the overlapping pixel patch, producing one output value. The resulting **feature map** highlights patterns the filter learned to detect (edges, textures). Multiple filters learn multiple patterns in parallel.

### Why is Pooling Used?
**Max-pooling** takes the maximum value within each small window (e.g. 2×2), reducing the feature map size. This:
1. Shrinks computation and parameters (speed + memory).
2. Provides **translation invariance** — the same feature is detected even if it moves a few pixels.
3. Discards low-confidence activations (noise suppression).

### Why is ReLU Commonly Used in CNNs?
ReLU (`f(x) = max(0, x)`) is the standard activation because:
- **Computationally cheap** — one comparison per neuron.
- **Avoids vanishing gradients** — gradient = 1 for positive inputs, unlike sigmoid/tanh which saturate.
- **Introduces non-linearity** — essential for learning complex, non-linear decision boundaries.

### Why Are CNNs Better than Feed-Forward Networks for Images?
A standard fully-connected network treats every pixel independently, ignoring spatial relationships and requiring enormous numbers of parameters. CNNs instead exploit **local connectivity** (filters look at small patches), **weight sharing** (the same filter is reused across all positions), and **hierarchical learning** (edges → shapes → objects through successive layers). This makes CNNs dramatically more parameter-efficient, robust to small translations, and accurate on vision tasks.


## Task 7 — Business Use Case: Manufacturing Quality Control

**Domain:** Automotive / Steel Parts Manufacturing

**Scenario:** A conveyor-belt camera captures images of every component surface. The CNN model classifies each surface in real time as normal or one of three defect types (scratch, dent, stain). Defective parts are automatically diverted for rework before final assembly.

**Business Impact:**

| Benefit | Detail |
|---|---|
| Cost savings | Defects caught early prevent expensive downstream rework or recalls |
| Speed | Thousands of inspections per hour vs. hundreds by human inspectors |
| Consistency | Same decision threshold applied 24/7; no fatigue or shift variation |
| Traceability | Every part ID linked to its classification result for audit compliance |
| Scalability | Model deployed across multiple production lines at near-zero marginal cost |

**Other applicable domains:**
- **Agriculture** — classifying fruit surface quality at packing stations.
- **Healthcare** — assisting dermatologists in skin lesion categorisation.
- **Retail e-commerce** — auto-tagging product condition (new, stained, scratched).
- **Security / Insurance** — automated vehicle damage assessment at car rentals or claim centres.

