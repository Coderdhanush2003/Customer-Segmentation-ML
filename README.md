# Customer Segmentation (ML)

This repository contains an end-to-end Machine Learning pipeline designed to explore, process, and segment a retail mall's customer base into distinct Categories.

The project leverages unsupervised machine learning techniques—specifically the K-Means clustering algorithm—paired with heuristic evaluation metrics to discover patterns in consumer spending and income profiles for target marketing optimizations.

## 📊 Dataset Overview

**Dataset Source:** (https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

The dataset tracks 200 individual customer records capturing basic demographic and economic profiles:
* **Demographic Metadata:** `CustomerID`, `Gender`, and `Age`.
* **Clustering Features (Predictors):** `Annual Income (k$)` and `Spending Score (1-100)` (a metric assigned by the mall based on customer behavior and purchasing history).

## 🛠️ Workflow & Code Structure

1. **Exploratory Data Analysis (EDA) & Data Cleaning:** Verifying dataset structure and ensuring integrity by checking for null entries using `df.isnull().sum()`.
2. **Feature Selection:** Extracting columns 3 and 4 (`Annual Income` and `Spending Score`) into an isolated feature matrix to map out clear purchasing relationships.
3. **The Elbow Method (Hyperparameter Tuning):** Iterating through cluster counts ($K = 1$ to $12$) to calculate the Within-Cluster Sum of Squares (WCSS). Plotting the inertias reveals an "elbow point" at $K = 6$, defining the optimal cluster capacity.
4. **Model Architecture & Fitting:** Initializing and training a `KMeans` algorithm set to 6 clusters with a deterministic `random_state=42` for consistent result replication.
5. **Data Mapping & Interpretation:** Appending structural cluster IDs back to the original dataframe and mapping numeric tags to explicit behavioral segment classifications.
6. **Data Visualization:** Rendering a clear multi-colored scatter plot mapping the 6 distinct clusters to visually demonstrate consumer boundaries.

### Key Takeaway:

The K-Means architecture successfully segments the 200-customer profile into 6 distinct behavioral tiers. The data points show that the vast majority of the customer base falls under **Average Customers** (77 counts), maintaining stable, mid-tier spending metrics. High-value targets are strictly isolated into **Premium Customers** (39 counts) who possess high incomes coupled with aggressive spending indexes, providing an immediate group for luxury promotional targeting.

## 🎯 Defined Customer Segments

| Cluster ID | Segment Name | Behavioral Cohort Profile | Distribution Count |
| :---: | :--- | :--- | :---: |
| **0** | **Average Customers** | Moderate Annual Income & Moderate Spending Score | 77 |
| **1** | **Premium Customers** | High Annual Income & High Spending Score | 39 |
| **2** | **Impulsive Spenders** | Low Annual Income & High Spending Score | 22 |
| **3** | **Careful Savers** | High Annual Income & Low Spending Score | 35 |
| **4** | **Moderate Spenders** | Low Annual Income & Low Spending Score | 12 |
| **5** | **Budget Customers** | Low/Mid Annual Income & Mid Spending Score | 15 |

## 🚀 Technologies Used

* **Python 3**
* **Pandas** (Data loading, filtering, feature isolation, and dictionary mapping)
* **Scikit-Learn** (KMeans model fitting, WCSS calculation, and metric definitions)
* **Matplotlib** (Elbow curve tracking and multi-cohort scatter plot visualizations)
* **Google Colab** (Interactive development and cloud runtime execution)
