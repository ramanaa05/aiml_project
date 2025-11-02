import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# --- FILE DEFINITIONS ---
FILE_NAME = 'featureEngineered.csv'
FINAL_OUTPUT_CSV = 'clustered_users_final.csv'

# --- 1. MODULARIZED FUNCTIONS ---

def load_and_preprocess_data(file_path):
    """Loads data, handles cleaning, imputation, encoding, and scaling."""
    df = pd.read_csv(file_path, encoding='unicode_escape')

    # Data Cleaning: Convert byte-strings (like b'female') to standard strings
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].astype(str).str.startswith("b'").any():
            df[col] = df[col].astype(str).str.replace("b'", "", regex=False).str.replace("'", "", regex=False)

    categorical_cols = ['gender', 'race', 'field']
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # a. Handle Missing Values (Imputation)
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # b. One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # c. Feature Scaling
    X = df_encoded.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df_encoded, scaler

def find_optimal_k(X_scaled, max_k=15):
    """Uses Silhouette Score to determine the optimal K."""
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
    plt.title('2. Elbow Method (Inertia)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.grid(True)
    plt.savefig('2_elbow_method_wcss.png')
    plt.close()

    # Plotting the Silhouette Score
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, marker='o', color='red')
    plt.title('3. Silhouette Scores for Cluster Selection')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.axvline(x=best_k_silhouette, color='green', linestyle='--', label=f'Optimal K={best_k_silhouette}')
    plt.legend()
    plt.grid(True)
    plt.savefig('3_silhouette_method.png')
    plt.close()

    return best_k_silhouette

def train_and_evaluate_model(X_scaled, df_encoded, optimal_k):
    """Trains the final model, adds labels, and reports metrics."""
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    cluster_labels = final_kmeans.fit_predict(X_scaled)

    # Add labels back to the DataFrame
    df_final = df_encoded.copy()
    df_final['Cluster_Label'] = cluster_labels

    # Evaluation Metrics
    final_inertia = final_kmeans.inertia_
    final_silhouette = silhouette_score(X_scaled, cluster_labels)

    print("\n--- Model Evaluation ---")
    print(f"K-Means Final Metrics (K={optimal_k}):")
    print(f"  Inertia (WCSS): {final_inertia:.2f}")
    print(f"  Silhouette Score: {final_silhouette:.4f}")

    return df_final, final_kmeans

def visualize_clusters(X_scaled, df_final, final_kmeans):
    """Reduces dimensionality using PCA and plots the clusters."""
    # Use PCA to reduce the high-dimensional data down to 2 components for plotting
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster_Label'] = df_final['Cluster_Label']

    # Plot the 2D visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        c=pca_df['Cluster_Label'],
        cmap='viridis',
        marker='o',
        s=50,
        alpha=0.6
    )

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

    plt.title('4. K-Means Clusters (2D PCA Projection)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('4_kmeans_scatter_plot.png')
    plt.close()

# --- MAIN EXECUTION ---
X_scaled, df_encoded, final_scaler = load_and_preprocess_data(FILE_NAME)
optimal_k = find_optimal_k(X_scaled)
df_final, final_kmeans = train_and_evaluate_model(X_scaled, df_encoded, optimal_k)
visualize_clusters(X_scaled, df_final, final_kmeans)

# Final Output (Prediction)
df_final.to_csv(FINAL_OUTPUT_CSV, index=False)
print(f"\nFinal clustered data saved to: {FINAL_OUTPUT_CSV}")

# Interpretation: Cluster Profiles
print("\n--- Cluster Interpretation (Centroids) ---")
cluster_profiles = df_final.groupby('Cluster_Label').mean()
print(cluster_profiles)