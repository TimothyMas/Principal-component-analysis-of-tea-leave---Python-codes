# Principal Component Analysis of Tea Leaves Chemical Ingredients (without data)

This repository contains Python scripts and resources for analyzing chemical components in tea leaves, using various statistical and data analysis techniques. The analysis aims to identify the type of tea based on chemical ingredient profiles using PCA and related statistical methods.

## Repository Overview

- **`p_IQR.py`**  
  Contains code for **outlier detection** using the IQR (Interquartile Range) method and related preprocessing steps to clean the data.

- **`p_intensity_for_all_matrix.py`**  
  Includes scripts for visualizing the **signal intensity distribution** and generating summary statistics such as median values and quantile ranges for chemical components.

- **`p_std_matrix.py`**  
  This file computes **standardized matrices** for PCA and other advanced statistical analyses. It includes scaling and transformation functions.

## Key Features

1. **Outlier Removal**:  
   Uses the IQR method to filter out extreme values, ensuring cleaner and more reliable data for further analysis.

2. **Signal Intensity Analysis**:  
   Provides visualizations and summary statistics of signal intensities across different compounds in tea.

3. **Principal Component Analysis (PCA)**:  
   Reduces dimensionality of the chemical ingredient dataset while highlighting key differences among tea types.

4. **Tea Type Classification**:  
   Maps chemical ingredient profiles to specific tea types (e.g., Black Tea, Green Tea, Pu-erh).

5. **Statistical Tests**:  
   Implements ANOVA, Kruskal-Wallis, and pairwise t-tests to assess significant differences among tea types.

6. **Log Transformation and Correlation Analysis**:  
   Applies log transformation to normalize data and computes correlation matrices for compounds.

## Data Analysis Workflow

1. **Preprocessing**:
   - Load and clean raw data.
   - Map EIC values to compounds and sample numbers to tea types.

2. **Outlier Detection**:
   - Identify and remove outliers using the IQR method.
   - Save outlier statistics for reference.

3. **Exploratory Analysis**:
   - Visualize data distributions using box plots, bar plots, and histograms.
   - Compute summary statistics (median, Q1, Q3, etc.).

4. **Dimensionality Reduction**:
   - Perform PCA to reduce the dataset to key components.
   - Visualize PCA results for better understanding of tea type clustering.

5. **Statistical Analysis**:
   - Conduct ANOVA and non-parametric tests to validate findings.
   - Generate correlation heatmaps to understand relationships among compounds.

## Visualization Examples

- **Bar Plots**:
  Median signal intensity for each compound grouped by tea type.
- **Box Plots**:
  Spread of signal intensities across tea types and compounds.
- **PCA Scatter Plots**:
  Visualization of tea type clustering in reduced dimensions.
- **Correlation Heatmaps**:
  Highlight relationships between compounds.

## Requirements

- Python 3.8+
- Required libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `sklearn`

Install the dependencies using:

```bash
pip install -r requirements.txt
