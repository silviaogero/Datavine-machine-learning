# DataVine Analytics: Machine Learning Applications

## Project Overview

**DataVine Analytics** is a boutique consulting group that develops data-driven solutions for organizations across different industries.

This project demonstrates the application of machine learning to three different business problems:

1. **Wine Classification**: Classifying wine varieties from chemical properties.
2. **Agricultural Feed Recommendation**: Identifying similar feed types based on chicken weight performance.
3. **Regional Crime Pattern Analysis**: Discovering natural groupings among regions based on crime statistics.

The project demonstrates three different machine learning approaches: **classification, similarity-based recommendation, and clustering**.

---

## Business Problems

### 1. Wine Classification System

A premium wine distributor wants to automatically classify wine varieties based on their chemical properties.

The project uses **k-Nearest Neighbors (k-NN)** combined with **Principal Component Analysis (PCA)**.

**Approach:**

* Standardize the chemical features.
* Apply PCA while retaining 95% of the variance.
* Use k-NN for classification.
* Use GridSearchCV to optimize the number of neighbors and distance metric.
* Evaluate the final model using accuracy, classification metrics, and a confusion matrix.

**Key Result:**

* Best parameters: `k = 18`, Euclidean distance
* Cross-validation accuracy: **97.91%**
* Test accuracy: **100%**

The results indicate that the chemical characteristics in the dataset provide strong information for distinguishing between the three wine classes.

---

### 2. Agricultural Feed Recommendation Engine

An agricultural supply company wants to identify feed types with similar performance characteristics.

The `Chickwts` dataset is used as a proxy for feed performance, with chicken weight serving as the performance measure.

**Approach:**

* Remove duplicate observations.
* Standardize chicken weight.
* Apply PCA to one component.
* Calculate cosine similarity.
* Aggregate similarity information by feed type.
* Generate similarity-based feed recommendations.

**Key Result:**

The analysis identified two broad groups of feeds with similar profiles in the one-dimensional PCA space:

* **Casein, meatmeal, and sunflower**
* **Horsebean, linseed, and soybean**

Because only one numerical feature was available, the cosine similarity results are relatively coarse. The recommendations should therefore be interpreted as a prototype similarity system rather than evidence that one feed causes better chicken growth.

---

### 3. Regional Crime Pattern Analysis

The objective of this project is to identify regions with similar crime patterns using the `USArrests` dataset.

The analysis focuses on:

* Murder
* Assault
* Rape

`UrbanPop` was excluded because it represents a demographic characteristic rather than a direct crime measure.

**Approach:**

* Select the relevant crime variables.
* Standardize the features.
* Apply PCA to reduce the data to two components.
* Use the K-Means elbow method to investigate the appropriate number of clusters.
* Use GMM BIC to investigate the appropriate number of Gaussian components.
* Compare the resulting cluster structures.
* Examine crime profiles within each cluster.

**Key Results:**

* PC1 explains **78.62%** of the variance.
* PC2 explains **15.27%** of the variance.
* The two components explain **93.89%** of the total variance.
* K-Means elbow analysis suggested **4 clusters**.
* GMM BIC suggested **2 components**.

GMM identified two broad regional profiles with relatively lower and higher average crime levels, while K-Means provided a more detailed segmentation of these patterns.

---

## Machine Learning Techniques

| Business Problem       | Techniques                              | Main Output                  |
| ---------------------- | --------------------------------------- | ---------------------------- |
| Wine Classification    | StandardScaler, PCA, k-NN, GridSearchCV | Wine class predictions       |
| Feed Recommendation    | StandardScaler, PCA, Cosine Similarity  | Similar feed recommendations |
| Crime Pattern Analysis | StandardScaler, PCA, K-Means, GMM       | Regional clusters            |

---

## Dataset Preparation

The datasets were inspected for:

* Missing values
* Duplicate records
* Data types
* Dataset dimensions
* Descriptive statistics
* Class and category distributions
* Potential data-quality issues

The datasets were generally complete.

The `Chickwts` dataset contained one duplicate observation, which was removed before analysis. The Wine and USArrests datasets did not contain duplicate records.

Feature scaling was performed separately within each project because the datasets use variables with different scales and units.

---

## Project Workflow

The overall machine learning workflow was:

```text
Understand Business Problem
          ↓
Explore Dataset
          ↓
Clean and Prepare Data
          ↓
Select Relevant Features
          ↓
Standardize Features
          ↓
Apply PCA Where Required
          ↓
Build Machine Learning Model
          ↓
Optimize / Determine Model Parameters
          ↓
Evaluate Results
          ↓
Visualize Findings
          ↓
Interpret Business Implications
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

### Scikit-learn Techniques

* `StandardScaler`
* `PCA`
* `KNeighborsClassifier`
* `GridSearchCV`
* `KMeans`
* `GaussianMixture`
* `cosine_similarity`
* Classification metrics

---

## Repository Structure

```text
DataVine-Analytics/
│
├── DataVine_Analytics.ipynb
├── wine.csv
├── chickwts.csv
├── USArrests.csv
└── README.md
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/DataVine-Analytics.git
```

### 2. Navigate to the project directory

```bash
cd DataVine-Analytics
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `DataVine_Analytics.ipynb` and run the notebook cells from top to bottom.

---

## Key Findings

The project demonstrates that different machine learning techniques are appropriate for different types of business problems.

* **k-NN** successfully classified wine varieties using chemical measurements.
* **PCA and cosine similarity** were used to create a prototype feed recommendation system.
* **K-Means and GMM** revealed different levels of structure within regional crime patterns.
* **PCA** provided dimensionality reduction while preserving most of the information in the datasets.
* **GridSearchCV** demonstrated how hyperparameter tuning can systematically improve model selection.

The project also highlights the importance of interpreting machine learning results within the limitations of the available data.

---

## Limitations

Several limitations should be considered when interpreting the results:

### Wine Classification

The 100% test accuracy was obtained on one held-out test set. It does not guarantee perfect performance on new wine samples outside the dataset.

### Feed Recommendation

The recommendation system uses only chicken weight as a performance measure. Because PCA is performed on a single feature, the resulting cosine similarity is limited and should be treated as a prototype rather than a production recommendation engine.

### Crime Pattern Analysis

The clustering results identify statistical patterns but do not establish causal relationships. Crime levels can be influenced by many social, economic, demographic, and institutional factors that are not included in the dataset.

---

## Conclusion

The DataVine Analytics project demonstrates an end-to-end machine learning workflow across three different analytical problems.

By combining data preparation, feature engineering, dimensionality reduction, supervised learning, similarity analysis, unsupervised learning, hyperparameter optimization, visualization, and business interpretation, the project shows how machine learning techniques can be adapted to different organizational needs.

The analysis emphasizes that successful machine learning is not only about building models, but also about selecting appropriate methods, evaluating their performance, understanding their limitations, and translating their results into meaningful business insights.
