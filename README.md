
# 🤝 Social Group Segmentation using Unsupervised Clustering

This repository contains the methodology, code, and artifacts for a project focused on segmenting user profiles into nuanced **social compatibility groups** using unsupervised clustering techniques. The goal is to move beyond simple binary matching (like dating) and provide a probabilistic profile suitable for flexible social recommendations.

## 1. Project Goal and Rationale

The primary objective is to define **five distinct social archetypes** based on users' interests, values, and personality scores.

* **Social Utility:** The project prioritizes **utility** ($K=5$ groups) over the dataset's naturally weak statistical structure ($K=2$ groups).
* **Methodological Choice:** **Fuzzy C-Means (FCM)** is the chosen final model because it delivers a **probabilistic output**, which accurately models the **blended social identity** of users (e.g., $65\%$ "Gamer" and $35\%$ "Art Enthusiast").

***

## 2. Dependencies and Setup

### 2.1 Required Libraries

The project requires the installation of the specialized `scikit-fuzzy` library for Fuzzy C-Means.

# Install standard libraries 
!pip install pandas numpy scikit-learn matplotlib

# Install the specialized Fuzzy C-Means library
!pip install scikit-fuzzy

### 2.2 Dataset Configuration

  * **Source File:** `featureEngineeredBetter.csv`
  * **Setup:** Upload this file to the root directory of your Google Colab or Jupyter environment.
  * **Feature Set:** The features are derived from the Speed Dating Experiment, including Age, Race, Field, and 24 self-rated scores (e.g., Sincerity, Ambition, Sports, Art). The **`gender`** attribute was intentionally **removed** from the features for all clustering models.

-----

## 3. How to Execute the Code

Execute the complete Python script file sequentially in your Colab notebook.

### 3.1 Model Configuration (Set in Code)

The model is configured to test the utility-driven structure required for the application:

| Variable | Description | Value |
| :--- | :--- | :--- |
| `FINAL_K` | The number of social groups to define (Utility-driven). | `5` |
| `FUZZINESS_M` | The fuzziness parameter. | `2.0` |

### 3.2 Methodological Blocks

The process follows a modular, sequential pipeline:

| Step | Block Name | Purpose |
| :--- | :--- | :--- |
| **Step 1** | **Pre-processing** | Loads data, handles **Imputation** (Median/Mode), applies **StandardScaler**, and runs **One-Hot Encoding** (expanding features to $\approx 300$ dimensions). |
| **Step 2** | **FCM Training** | Runs the Fuzzy C-Means algorithm to find cluster **Centroids** and the **Membership Matrix ($\mathbf{U}$)**. |
| **Step 3** | **Evaluation & PCA** | Calculates the **Fuzzy Partition Coefficient (FPC)**. Uses **PCA** for dimensionality reduction (2D) to enable visualization. |
| **Step 4** | **Output** | Generates the final CSV file with membership scores and the visualization plot. |

-----

## 4. Output Artifacts and Interpretation

The execution generates the following files and provides the critical model outputs for social group deployment.

### 4.1 Prediction Output

  * **File Name:** `social_clustered_users_fuzzy.csv`
  * **Contents:** The file contains all original features plus five new columns: `Membership_Group_0` through `Membership_Group_4`.
  * **Inference:** This is the most crucial output. For any user, the values in these five columns represent their probability profile (e.g., User A is $70\%$ Group 2, $15\%$ Group 4). Recommendations should be sourced from the groups where a user has high membership probability.

### 4.2 Visualization and Diagnostics

| Output Artifact | Type | Interpretation |
| :--- | :--- | :--- |
| `fuzzy_cmeans_scatter_plot.png` | **Plot** | A 2D visualization of the 5 groups (colored by highest membership). The reader should infer that the visual separation is an **oversimplification**; the clusters overlap heavily in the full 300-dimensional space, which is why the soft FCM approach is necessary. |
| **FCM Centroids (Console Output)** | **Model Interpretation** | These are the $\mathbf{5}$ **average profile vectors** for each social group. Comparing the scores (e.g., high `sports` vs. low `art`) defines the persona of each social archetype (e.g., "The Active Socializer" vs. "The Homebody Reader"). |
| **Fuzzy Partition Coefficient (FPC)** | **Metric** | A value close to 1 confirms that users have **high membership in multiple groups**, validating the fuzzy approach for blended social identities. |
