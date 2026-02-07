
---

# PCA Analysis of Cervical Cancer Risk Factors Dataset

**Python NumPy Pandas Matplotlib**

---

## Overview

This project demonstrates the application of Principal Component Analysis (PCA) on the **Cervical Cancer Risk Factors Dataset** from the UCI Machine Learning Repository. PCA is a statistical technique used to emphasize variation and bring out strong patterns in a dataset. It helps in reducing the dimensionality of the data while retaining most of the variance, making it easier to visualize and analyze complex datasets.

---

## Dataset

* **Name:** Cervical Cancer Risk Factors Dataset
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors)
* **Description:** Contains medical and behavioral risk factors, as well as demographic information, related to cervical cancer. Includes numeric and categorical features, some with missing values.

---

## Features

* **Data Cleaning & Preprocessing:**
  Handles missing values, encodes categorical variables, and ensures numeric features are ready for PCA.

* **Data Standardization:**
  Scales numeric features to mean 0 and variance 1 to avoid bias due to different feature ranges.

* **PCA Computation:**
  Computes covariance matrix, eigenvalues, and eigenvectors to determine principal components.

* **Explained Variance Analysis:**
  Calculates variance ratios and cumulative variance to quantify the amount of information retained by each principal component.

* **Visualization:**

  * Cumulative explained variance curve
  * Original vs PCA-reduced data scatter plots

---

## Installation

```bash
# Clone repository
git clone https://github.com/Mukunzijames/CC-Principal-component-analysis.git
cd CC-Principal-component-analysis

# Install dependencies
pip install numpy pandas matplotlib
```

---

## Quick Start Code Example

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
df = pd.read_csv(url)

# Convert '?' to NaN and numeric columns
df.replace("?", np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Drop rows with missing values (or handle them as needed)
df_clean = df.dropna()

# Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean)

# Apply PCA
pca = PCA(n_components=2)  # Example: first 2 components
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Plot PCA-reduced data
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Cervical Cancer Risk Factors Dataset')
plt.show()
```

---

## Usage

1. Load the dataset using Pandas.
2. Preprocess the data: handle missing values, encode categorical variables, and standardize numeric features.
3. Apply PCA: compute covariance, eigenvalues/eigenvectors, and sort components by explained variance.
4. Visualize results:

   * Original vs PCA-reduced scatter plots
   * Cumulative explained variance curve
5. Analyze explained variance to decide how many components to retain for modeling or visualization.

---

## Interpretation Example

* **PC1:** captures 14.99% of total variance.
* **PC2:** captures 9.89% variance, orthogonal to PC1.
* **Cumulative variance:** guides how many components are needed to retain desired information (e.g., 90%).

PCA reduces feature redundancy while highlighting the main patterns and relationships in the cervical cancer risk factors dataset.

---

## Author

Mukunzi James – [mukunzindahiro@gmail.com](mailto:mukunzindahiro@gmail.com)

---