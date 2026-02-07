
---

# PCA Analysis of African Malaria Synthetic Dataset

**Python · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn**

---

## Overview

This project demonstrates **Principal Component Analysis (PCA)** applied to the **African Malaria Synthetic Dataset** (baseline, 1,000 observations) from Hugging Face. PCA is implemented from scratch to identify main patterns in malaria-related features in Sub-Saharan Africa. The notebook is a formative assignment in Advanced Linear Algebra, covering data loading, standardization, covariance-based PCA, explained variance analysis, and visualization.

---

## Dataset

* **Name:** African Malaria Synthetic Dataset – baseline sample (1,000 observations)
* **Source:** [Hugging Face – electricsheepafrica/african-malaria-dataset](https://huggingface.co/datasets/electricsheepafrica/african-malaria-dataset)
* **Description:** Synthetic dataset simulating demographic, clinical, and epidemiological features relevant to malaria in Sub-Saharan Africa. Includes age, sex, residence, season, fever, parasitemia, anemia, severe outcomes (e.g. cerebral malaria, respiratory distress), and malaria probability score. Contains numeric, categorical, and boolean variables, with some missing values (e.g. in `anemia_status`, `parasitemia_level`), making it suitable for preprocessing, dimensionality reduction, and PCA.

---

## Features

* **Data loading & exploration:** Load CSV from Hugging Face, inspect shape, dtypes, and missing values.
* **Data cleaning & preprocessing:** Handle missing values, encode categorical/boolean variables, and prepare numeric features for PCA.
* **Standardization:** Scale numeric features to mean 0 and standard deviation 1 (formula-based implementation as in the assignment).
* **PCA from scratch:** Compute covariance matrix, eigenvalues, and eigenvectors; sort components by explained variance.
* **Explained variance analysis:** Variance ratio and cumulative variance per component; choice of number of components (e.g. 90% threshold).
* **Visualization:**
  * Cumulative explained variance curve (with threshold marker)
  * Scatter: original feature space vs PCA-reduced (PC1 vs PC2)
  * Bar chart of PC1 loadings (feature contributions to main pattern)
  * Optional: scatter colored by PC1 score
* **Validation:** Orthogonality of principal components (dot products ≈ 0); comparison with scikit-learn PCA (e.g. explained variance, PC1 correlation).

---

## Installation

```bash
# Clone repository
git clone https://github.com/Mukunzijames/CC-Principal-component-analysis.git
cd CC-Principal-component-analysis

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Quick Start Code Example

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load African Malaria baseline dataset
url = "https://huggingface.co/datasets/electricsheepafrica/african-malaria-dataset/resolve/main/malaria_ssa_baseline_1000.csv"
df = pd.read_csv(url)

# Preprocess: select numeric columns, handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols].dropna(axis=0)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# PCA (e.g. first 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)

# Plot PCA-reduced data
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of African Malaria Synthetic Dataset')
plt.show()
```

---

## Usage

1. Open `Mukunzi_James_PCA_Formative_1.ipynb` and run cells in order.
2. Load the dataset from the URL above; preprocess (handle missing values, encode categoricals, keep numeric features for PCA).
3. Standardize numeric features (mean 0, std 1).
4. Apply PCA from scratch: covariance matrix → eigenvalues/eigenvectors → sort by variance → project data.
5. Visualize: cumulative variance curve, PC1 vs PC2 scatter, PC1 loadings bar chart.
6. Use explained variance to decide how many components to retain (e.g. 90% cumulative variance).

---

## Interpretation (from notebook)

* **PC1** captures about **39.43%** of total variance and represents the dominant malaria risk/severity pattern (e.g. malaria probability score, parasitemia, fever, anemia-related features).
* **PC2** captures about **22.80%** variance (orthogonal to PC1), reflecting secondary risk-factor combinations.
* **Cumulative variance:** First two PCs explain ~**62.23%**; first three and five PCs capture higher cumulative shares. The cumulative curve guides how many components to keep for a desired retention (e.g. 90%).

PCA reduces dimensionality and redundancy while highlighting the main patterns in the African malaria synthetic dataset.

---

## Author

Mukunzi James – [mukunzindahiro@gmail.com](mailto:mukunzindahiro@gmail.com)

---
