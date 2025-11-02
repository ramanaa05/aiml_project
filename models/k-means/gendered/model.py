import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# --- FILE AND MODEL CONFIGURATION ---
FILE_NAME = 'featureEngineeredBetter.csv'
FINAL_OUTPUT_CSV = 'social_clustered_users_k5_final.csv'
FINAL_K = 5

# --- 1. MODULARIZED FUNCTIONS ---

def load_and_preprocess_data(file_path):
    """Loads data, handles cleaning, imputation, encoding, and scaling."""
    print("--- 1. Data Preprocessing Started (Skipping header row 0) ---")

    # FIX: Use header=1 to correctly read the CSV, skipping the first row
    df = pd.read_csv(file_path, encoding='unicode_escape', header=1)

    # Data Cleaning: Convert byte-strings (like b'...') to standard strings
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].astype(str).str.startswith("b'").any():
            df[col] = df[col].astype(str).str.replace("b'", "", regex=False).str.replace("'", "", regex=False)

    # Define features for clustering (Gender is already excluded)
    categorical_cols = ['race', 'field']
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # a. Handle Missing Values (Imputation)
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # b. One-Hot Encoding for categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Robustness Check
    if df_encoded.empty or df_encoded.shape[1] == 0:
        raise ValueError("DataFrame is empty or has zero columns after preprocessing. Check file content.")

    # c. Feature Scaling: Standardize all features
    X = df_encoded.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Features used for clustering: {X_scaled.shape[1]}")
    print("--- 1. Data Preprocessing Complete ---")
    return X_scaled, df_encoded, scaler

def find_optimal_k_and_plot(X_scaled, max_k=15):
    """Uses Silhouette Score to determine the optimal K and plots WCSS/Silhouette."""
    # (Function body for K analysis remains unchanged)
    print("--- 2. Analyzing K (Elbow & Silhouette) Started ---")

    inertia_values = []
    silhouette_scores = []
    k_range = range(2, max_k)

    # K=1 for WCSS plot
    kmeans_k1 = KMeans(n_clusters=1, random_state=42, n_init='auto').fit(X_scaled)
    inertia_values.append(kmeans_k1.inertia_)
    k_range_plot = range(1, max_k)

    best_k_silhouette = 2
    max_silhouette = -1

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        inertia_values.append(kmeans.inertia_)

        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)

        if score > max_silhouette:
            max_silhouette = score
            best_k_silhouette = k

    # Plotting the Elbow Curve (WCSS)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range_plot, inertia_values, marker='o')
    plt.title('2. Elbow Method (Inertia) - New Feature Set')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.grid(True)
    plt.savefig('2_new_elbow_method_wcss.png')
    plt.close()

    # Plotting the Silhouette Score
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o', color='red')
    plt.title('3. Silhouette Scores - New Feature Set')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.axvline(x=best_k_silhouette, color='green', linestyle='--', label=f'Best K={best_k_silhouette} (Max Score)')
    plt.legend()
    plt.grid(True)
    plt.savefig('3_new_silhouette_method.png')
    plt.close()

    print(f"Mathematically Optimal K (Silhouette Score): {best_k_silhouette}")
    print("--- 2. K Analysis Complete (New Plots Saved) ---")
    return best_k_silhouette

def train_and_evaluate_model(X_scaled, df_encoded, final_k):
    """Trains the final model using the fixed K, adds labels, and reports metrics."""
    # (Function body remains unchanged)
    print(f"--- 3. Final Model Training Started (K={final_k}) ---")

    final_kmeans = KMeans(n_clusters=final_k, random_state=42, n_init='auto')
    cluster_labels = final_kmeans.fit_predict(X_scaled)

    df_final = df_encoded.copy()
    df_final['Social_Group_Label'] = cluster_labels

    final_inertia = final_kmeans.inertia_
    final_silhouette = silhouette_score(X_scaled, cluster_labels)

    print("\n--- Model Evaluation ---")
    print(f"K-Means Final Metrics (K={final_k}):")
    print(f"  Inertia (WCSS): {final_inertia:.2f}")
    print(f"  Silhouette Score: {final_silhouette:.4f}")

    print("Cluster Sizes:")
    print(df_final['Social_Group_Label'].value_counts().sort_index())

    print("--- 3. Final Model Training & Evaluation Complete ---")
    return df_final, final_kmeans

def visualize_clusters(X_scaled, df_final, final_kmeans, final_k):
    """Reduces dimensionality using PCA and plots the clusters with distinct colors."""
    print("--- 4. Cluster Visualization Started ---")

    # Use PCA to reduce the data down to 2 components for plotting
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Social_Group_Label'] = df_final['Social_Group_Label']

    plt.figure(figsize=(12, 8))

    # *** MODIFICATION START ***
    # Use 'tab10' for distinct categorical colors for up to 10 clusters
    plt.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        c=pca_df['Social_Group_Label'],
        cmap='tab10',  # <-- Changed cmap from 'viridis' to 'tab10'
        marker='o',
        s=50,
        alpha=0.6
    )
    # *** MODIFICATION END ***

    # Plot the cluster centers (Centroids)
    centroids_pca = pca.transform(final_kmeans.cluster_centers_)
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker='X',
        s=200,
        color='red',
        label='Centroids'
    )

    plt.title(f'4. K-Means Social Groups (K={final_k} | 2D PCA Projection)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('4_new_kmeans_scatter_plot.png')
    plt.close()

    print("--- 4. Cluster Visualization Complete (New Plot Saved) ---")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # a. Preprocessing
    try:
        X_scaled, df_encoded, scaler = load_and_preprocess_data(FILE_NAME)
    except Exception as e:
        print(f"\nFATAL ERROR DURING PREPROCESSING: {e}")
        print("Please check file path, encoding, and column names.")
    else:
        # b. Find Optimal K (for comparison/justification)
        find_optimal_k_and_plot(X_scaled)

        # c. Train and Evaluate Final Model (using fixed K=5 for social utility)
        df_final, final_kmeans = train_and_evaluate_model(X_scaled, df_encoded, FINAL_K)

        # d. Visualize and Save
        visualize_clusters(X_scaled, df_final, final_kmeans, FINAL_K)

        # e. Save Final Output
        df_final.to_csv(FINAL_OUTPUT_CSV, index=False)

        print(f"\nFinal clustered data saved to: {FINAL_OUTPUT_CSV}")
        print("\n--- Cluster Interpretation (Centroids) ---")
        cluster_profiles = df_final.groupby('Social_Group_Label').mean()
        print(cluster_profiles)