# Remote Sensing Landcover Classification: Random Forest vs CNN

## Task Description

This project explores landcover classification from remote sensing images using two machine learning approaches: a Random Forest classifier (scikit-learn) and a Convolutional Neural Network (CNN) (PyTorch). The goal is to classify 28×28 pixel satellite image patches into one of six landcover classes, compare the models, and investigate the effect of different input channels and hyperparameters.

This work is part of the MLESS SoSe 2026 homework assignment by Prof. Dr. Martin Schultz, University of Cologne.

## Dataset

The data is a subset of the [SAT-6 dataset](https://csc.lsu.edu/~saikat/deepsat/) (Basu et al., ACM SIGSPATIAL 2015), hosted on [B2SHARE](https://b2share.eudat.eu/records/89654eac10724d30a6c7e51f2c5422de).

- **81,000 samples** (28×28 pixels, 4 channels: R, G, B, NIR)
- **6 classes**: building, barren_land, trees, grassland, road, water
- Labels are one-hot encoded
- For experiments: 1,000 training samples and 100 test samples per class

The three CSV files (`X_test_sat6.csv`, `y_test_sat6.csv`, `sat6annotations.csv`) must be placed in a `data/` directory.

## Setup and Requirements

```bash
pip install -r requirements.txt
```

**Python version:** 3.11+ recommended

**Note:** On Windows, the `wget` cells in the notebooks will not work. Download the data files manually from the B2SHARE link above and place them in the `data/` folder.

## Methods and Results

### Task 2: Random Forest Classifier

#### 2.1: Baseline and Notebook Questions

The Random Forest classifier was trained with `n_estimators=100` (default settings) on all 4 channels (R, G, B, NIR).

**Baseline accuracy: 0.943**

**Answers to notebook questions:**

**Q: What would you need to do to extract only the green and infrared channel?**
Each image is stored as a flat row of 3,136 values (4 channels × 28 × 28 = 3,136). The channel order is R, G, B, NIR, with 784 values per channel. To extract only green and NIR, select columns 784–1,567 (G) and 2,352–3,135 (NIR).

**Q: What is the advantage of one-hot encoding compared to simple class labels?**
One-hot encoding avoids implying an ordinal relationship between classes. With integer labels (0, 1, 2, ...), a model might interpret class 3 as "greater than" class 1. One-hot encoding treats all classes as equally distinct. It is also the standard output format for neural network classifiers and supports multi-label classification.

**Q: Why use `extend` here and `append` above?**
`append` adds a list as a single nested element (creating a list of lists), which was needed above to keep samples grouped by class. `extend` adds each element individually into a flat list, which is needed here to create one continuous list of all training/test indices.

**Q: What is wrong with the above code?**
The training and test indices are sampled independently from the same pool, so there may be overlap between training and test sets. This could lead to artificially inflated accuracy, since the model may be tested on samples it has already seen during training.

**Q: Why shuffle the samples in train and test datasets?**
Without shuffling, all samples of class 0 appear first, then class 1, etc. This sequential ordering can introduce bias during training. Shuffling ensures the model sees a random mix of classes, leading to better generalization.

#### 2.2: Per-Class Accuracy

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| building    | 0.82      | 0.97   | 0.89     |
| barren_land | 0.99      | 0.92   | 0.95     |
| trees       | 0.97      | 0.95   | 0.96     |
| grassland   | 0.94      | 0.89   | 0.91     |
| road        | 0.97      | 0.93   | 0.95     |
| water       | 1.00      | 1.00   | 1.00     |

Water achieves perfect classification. Grassland and building are the most challenging classes, likely due to visual similarity with other classes.

#### 2.3: RGB Only (no NIR)

**Accuracy: 0.947**

Removing the NIR channel did not degrade performance. The accuracy is nearly identical to the 4-channel baseline (0.943). This suggests that for the Random Forest, the RGB channels already contain sufficient information for classification, and the NIR channel provides little additional discriminative power.

#### 2.4: R, G, NIR (no Blue)

**Accuracy: 0.947**

Again, performance is unchanged compared to the baseline. The Random Forest appears robust to dropping any single channel, indicating that the remaining three channels carry enough information.

#### 2.5: Hyperparameter Experiment — `max_depth`

**Hyperparameter chosen:** `max_depth` (default: None/unlimited → changed to: 10)

**Justification:** With unlimited depth, individual trees can grow very deep and memorize the training data (overfitting). Limiting the depth to 10 forces the trees to learn more general patterns rather than noise.

**Expectation:** Accuracy might decrease slightly because the trees are less expressive, but the model should be more robust (less overfitting) and generalize better to unseen data.

**Result:** Accuracy dropped slightly from 0.943 to **0.938**.

**Discussion:** The result matches the expectation. The small decrease (0.5%) indicates that the baseline model was not heavily overfitting, so constraining the depth had only a minor effect. In a scenario with noisier data, limiting depth could actually improve test accuracy by preventing overfitting.

### Task 3: CNN Classifier

#### 3.1: Baseline

The CNN architecture consists of 3 convolutional layers (32, 64, 128 filters) with kernel sizes (5, 3, 3), average pooling, a fully connected layer of 32 units, and ReLU activation. Training used Adam optimizer (lr=0.001), CrossEntropy loss, batch size 256, and 10 epochs.

**Baseline accuracy: 0.958**

The CNN outperforms the Random Forest (0.958 vs 0.943), demonstrating that the CNN can better exploit the spatial structure of the image data.

#### 3.2: Per-Class Accuracy

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| building    | 0.91      | 0.98   | 0.94     |
| barren_land | 0.99      | 0.95   | 0.97     |
| trees       | 0.97      | 0.97   | 0.97     |
| grassland   | 0.92      | 0.97   | 0.95     |
| road        | 0.98      | 0.88   | 0.93     |
| water       | 0.99      | 1.00   | 1.00     |

**What changed from Task 2.2:**
- In the Random Forest, predictions are obtained via `rf.predict()`. In the CNN, predictions (`pred_labels`) and true labels (`true_labels`) are already collected during the training/evaluation loop.
- In the Random Forest, labels are one-hot encoded arrays. In the CNN, they are integer class indices (converted from one-hot in the Dataset class).
- The plotting logic is similar, but the data reshaping differs: CNN test data needs to be reshaped from the flat dataframe back to (28, 28, 4) for visualization.

#### 3.3: CNN with RGB Only

**Accuracy: 0.937** (down from 0.958)

The CNN suffered a larger drop (−2.1%) compared to the Random Forest (−0.0% change) when removing the NIR channel. This indicates that the CNN was more effectively leveraging the NIR information through its convolutional filters. The Random Forest, treating each pixel independently, did not benefit as much from the NIR channel.

| Model         | 4-Channel | RGB Only | Change |
|---------------|-----------|----------|--------|
| Random Forest | 0.943     | 0.947    | +0.4%  |
| CNN           | 0.958     | 0.937    | −2.1%  |

#### 3.4: Hyperparameter Experiment — Learning Rate

**Hyperparameter chosen:** Learning rate (default: 0.001 → changed to: 0.0001)

**Justification:** The learning rate controls the step size during gradient descent. A smaller learning rate leads to more precise convergence but requires more training epochs. With a fixed budget of 10 epochs, a much smaller learning rate may not allow the model to converge.

**Expectation:** Accuracy will decrease because 10 epochs is insufficient for the model to converge with such a small learning rate. The model would be more stable (less risk of overshooting optima) but will underfit given the limited training time.

**Result:** Accuracy dropped significantly from 0.958 to **0.850**.

**Discussion:** The result strongly confirms the expectation. The road class was hit hardest (0.88 → 0.63), while water remained perfect (1.00). This demonstrates that hyperparameter choices interact — the learning rate and number of epochs must be tuned together. To benefit from a smaller learning rate, significantly more epochs would be needed.

## Summary of Results

| Experiment                    | RF Accuracy | CNN Accuracy |
|-------------------------------|-------------|--------------|
| Baseline (4 channels)         | 0.943       | 0.958        |
| RGB only                      | 0.947       | 0.937        |
| R, G, NIR (RF only)           | 0.947       | —            |
| Hyperparameter change         | 0.938       | 0.850        |

Key findings:
1. The CNN outperforms the Random Forest on the baseline task (95.8% vs 94.3%).
2. The CNN benefits more from the NIR channel than the Random Forest.
3. Both models are sensitive to hyperparameter choices, but the CNN shows larger performance swings.

## Repository Structure

```
remote-sensing-lc/
├── data/                          # Dataset files (not tracked by git)
│   ├── X_test_sat6.csv
│   ├── y_test_sat6.csv
│   └── sat6annotations.csv
├── Random_forest_classifier_on_remote_sensing_image.ipynb
├── CNN_classifier_on_remote_sensing_image.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## License

MIT License

## Author

Mohammed Fawaz Nawaz,
Msc, Computational sciences
University of Cologne
