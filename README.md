# Remote sensing for landcover detection: ML techniques

This repository contain course material for the MLESS lecture by Prof. Dr. Martin Schultz at the University of Cologne, Germany.
Two Jupyter notebooks demonstrate the use of random forests and convolutional neural networks for a simplified landcover classification task
based on satellite remote sensing data.

The data for these demonstrations is a subset from the [SAT-6](https://csc.lsu.edu/~saikat/deepsat/) dataset by
_Saikat Basu, Sangram Ganguly, Supratik Mukhopadhyay, Robert Dibiano, Manohar Karki and Ramakrishna Nemani, DeepSat - A Learning framework for Satellite Imagery, ACM SIGSPATIAL 2015._
It consists of 28x28 pixels uint8 images with 4 channels and labels of 6 landcover classes - barren land, trees, grassland, roads, buildings and water bodies.

The classification task is scene classification, i.e. the entire 28x28 pixel image is classified a sone landcover type.

## Download the data

Unfortunately, there is no anonymous and free access to the Deepsat data. However, a subset has been extracted and made available atthe B2SHARE server at FZ Jülich: 
https://b2share.eudat.eu/records/89654eac10724d30a6c7e51f2c5422de. Thi scomprises only the test set of the original data - for our educational experiments, this is sufficient.

The three data files must be stored in a `data` directory in the same path as the notebook itself.

## Run the example notebooks

Start with the Random_forest_classifier notebook. WARNING: loading the data into pandas consumes ~5 GBytes of memory. Make sure that your Jupyter lab has sufficient memory.

Once you fully understood what this notebook does, take a look at the CNN_classifier notebook and run it. Note that you need to have Pytorch installed to run the CNN_classifier notebook.

Compare training times, inference times (if you notice a difference) and the quality of the results.

Think about the network and training paraneters: which ones would you modify if you want to improve the results?

---

# Project Extensions

The original Random Forest notebook was extended with the following functionality:

* Function-based experiment framework
* Balanced train/test sampling
* Automatic channel selection
* NDVI computation and integration
* Evaluation of different spectral channel combinations
* Misclassification analysis
* Experiment result tracking and comparison

---
# NDVI Calculation

NDVI combines information from the red and near-infrared channels and highlights vegetation-covered regions. Vegetation reflects strongly in the NIR spectrum while absorbing visible red light. Therefore, NDVI is particularly useful for distinguishing vegetation from other land cover classes.

To investigate the influence of vegetation information, the Normalized Difference Vegetation Index (NDVI) was calculated:

$$
NDVI = \frac{NIR - Red}{NIR + Red}
$$

The NDVI layer was added as a fifth image channel: ``` (R, G, B, NIR, NDVI) ``` This increases the number of features from 3136 to 3920 per image.

---

# Experimental Setup

A balanced dataset was used throughout all experiments. For each land cover class:

* 1000 samples were selected for training
* 100 samples were selected for testing

Resulting in 6000 training samples, 600 test samples.

The Random Forest classifier was trained using:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

---

# Questions and Answers

## What would you need to do to extract only the green and the infrared channel from this data? 

One sample in the data has format RGBNIR,RGBNIR,... or in matrix form: $$ 28 \times 28 \times 4 = 3136 $$, hence use G = img[:,:,1] for green.

## What is the advantage of this encoding compared to a simple class label like '0', '1', '2', '3', '4', '5', or text labels like 'building', 'barren_land', ...?

ML algorithms operate on numerical data, so class labels must be encoded numerically rather than as text. Numerical encoding is also more memory-efficient and computationally faster than string labels. One-hot encoding prevents the model from interpreting class labels as ordinal values. For example, labels 0,1,2,3,4,5 may incorrectly imply that class 5 is “larger” or “closer” to class 4 than to class 1, although the classes are actually categorical and unordered.

## Why use `extend`here and `append` above? 

To have as a result a vector, not a nested array

## What is wrong with the above code? 

There is an overlap possible for a data between the train and test samples -> consider overfitting.

## Why do you want to shuffle the samples in the train and test datasets?

Shuffling removes this ordering and ensures a random distribution of classes throughout the training and testing datasets. This prevents unintended ordering effects and produces a more realistic ML experiment.

## Why use a balanced dataset?

The original SAT-6 dataset contains different numbers of samples for different classes. Using an equal number of samples per class prevents the classifier from becoming biased towards classes with more training examples.

## Why is NDVI useful?


---

# Results

The table below summarizes the classification accuracy and training time obtained for different channel combinations.

| Channels         | Features |   Accuracy | Training Time (s) |
| ---------------- | -------: | ---------: | ----------------: |
| RGB + NIR + NDVI |     3920 | **0.9433** |             36.99 |
| RGB + NIR        |     3136 |     0.9300 |             33.74 |
| RG + NIR         |     2352 |     0.9200 |             25.59 |
| RGB              |     2352 |     0.8883 |             28.16 |
| NDVI             |      784 | **0.8483** |             32.57 |
| R                |      784 |     0.7083 |             25.36 |
| NIR              |      784 |     0.6800 |             22.44 |
| G                |      784 |     0.6517 |             21.46 |
| B                |      784 |     0.6233 |             19.89 |

---

# Key Findings

* The highest classification accuracy (**94.33%**) was achieved using **RGB + NIR + NDVI**.
* Adding the NIR channel improved the accuracy from **88.83%** (RGB) to **93.00%**.
* Adding NDVI further increased the accuracy to **94.33%**.
* NDVI was the best individual channel, achieving **84.83%** accuracy.
* NDVI significantly outperformed all individual spectral bands (R, G, B, and NIR).
* The blue channel produced the lowest classification accuracy (**62.33%**).
* Training times generally increased with the number of input features.

---

# Discussion

The experiments clearly demonstrate the importance of spectral information in remote sensing applications.

RGB channels alone already provide a strong baseline performance of 88.83%. The addition of the NIR channel significantly improves classification accuracy, reflecting the importance of near-infrared reflectance for distinguishing land cover types.

The highest performance was obtained by including NDVI as an additional feature. This result is expected because NDVI captures vegetation-related information that is not directly available from the original spectral channels.

An interesting observation is that NDVI alone achieved an accuracy of 84.83%, substantially higher than any individual spectral band. This indicates that vegetation information is one of the most informative features for land cover classification in the SAT-6 dataset.

Overall, the combination of RGB, NIR, and NDVI produced the most accurate classification model.

---

# Possible Improvements

Several Random Forest hyperparameters could be investigated further:

* Number of trees (`n_estimators`)
* Maximum tree depth (`max_depth`)
* Number of features considered per split (`max_features`)
* Minimum number of samples required for splitting (`min_samples_split`)
* Minimum number of samples per leaf (`min_samples_leaf`)

Additional future work could include:

* Larger balanced training datasets
* Additional vegetation indices
* Hyperparameter optimization
* Comparison with CNN-based approaches
* Cross-validation experiments

---

# Author

Donata Banyte

University of Cologne

MLESS – Machine Learning in Earth System Science
