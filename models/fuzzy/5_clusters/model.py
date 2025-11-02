import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import skfuzzy as fuzz # NOTE: Requires installation: !pip install scikit-fuzzy
from sklearn.metrics import silhouette_score # Used for evaluating hard assignments

# --- FILE AND MODEL CONFIGURATION ---
FILE_NAME = 'featureEngineeredBetter.csv'
FINAL_OUTPUT_CSV = 'social_clustered_users_fuzzy.csv'
FINAL_K = 5  # Using K=5 for distinct social groups
FUZZINESS_M = 2.0 # Fuzziness parameter (m > 1). Higher m means fuzzier clusters.

# --- 1. MODULARIZED FUNCTIONS ---

def load_and_preprocess_data(file_path):
    """Loads data, handles cleaning, imputation, encoding, and scaling."""
    print("--- 1. Data Preprocessing Started ---")

    # Use header=1 to correctly read the CSV, skipping the first row
    df = pd.read_csv(file_path, encoding='unicode_escape', header=1)

    # Data Cleaning: Convert byte-strings to standard strings
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

    # Feature Scaling: Standardize all features
    X = df_encoded.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Features used for clustering: {X_scaled.shape[1]}")
    print("--- 1. Data Preprocessing Complete ---")
    return X_scaled, df_encoded, scaler

def train_fuzzy_c_means(X_scaled, final_k, fuzziness_m):
    """Trains the Fuzzy C-Means model and returns centroids and membership matrix."""
    print(f"--- 2. Fuzzy C-Means Training Started (K={final_k}, m={fuzziness_m}) ---")

    # Transpose X_scaled because skfuzzy expects features on the rows (N_features x N_samples)
    X_fcm = X_scaled.T

    # Train FCM: The output `u` is the membership matrix
    # The output `cntr` is the cluster centers (centroids)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_fcm,
        c=final_k,
        m=fuzziness_m,
        error=0.005,
        maxiter=1000,
        seed=42
    )

    print(f"Final Fuzzy Partition Coefficient (FPC): {fpc:.4f}")
    print("--- 2. Fuzzy C-Means Training Complete ---")
    return cntr, u, fpc

def evaluate_and_save_fuzzy_results(X_scaled, df_encoded, cntr, u, final_k):
    """Evaluates the fuzzy model and saves the membership matrix."""

    # 1. Save Membership Scores
    u_df = pd.DataFrame(u.T, columns=[f'Membership_Group_{i}' for i in range(final_k)])
    df_final = pd.concat([df_encoded.reset_index(drop=True), u_df], axis=1)

    # 2. Get Hard Cluster Labels (for visualization/silhouette score)
    # The hard label is the group where the user has the highest membership score
    hard_labels = u.T.argmax(axis=1)
    df_final['Hard_Label'] = hard_labels

    # 3. Calculate Silhouette Score on Hard Labels (as a proxy for separation)
    silhouette_avg = silhouette_score(X_scaled, hard_labels)

    print("\n--- Model Evaluation ---")
    print(f"Hard-Assigned Silhouette Score: {silhouette_avg:.4f}")
    print("Cluster Sizes (Based on Hard Assignment):")
    print(df_final['Hard_Label'].value_counts().sort_index())

    df_final.to_csv(FINAL_OUTPUT_CSV, index=False)
    print(f"\nFinal fuzzy data (with membership scores) saved to: {FINAL_OUTPUT_CSV}")

    return df_final, hard_labels, cntr

def visualize_fuzzy_clusters(X_scaled, hard_labels, final_k, cntr):
    """Reduces dimensionality and plots the clusters based on hard assignment."""
    print("--- 3. Cluster Visualization Started ---")

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Hard_Label'] = hard_labels

    plt.figure(figsize=(12, 8))

    # Plotting points based on hard assignment labels
    plt.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        c=pca_df['Hard_Label'],
        cmap='tab10',
        marker='o',
        s=50,
        alpha=0.6
    )

    # Plot the cluster centers (Centroids)
    # Centroids need to be transformed by PCA as well
    centroids_pca = pca.transform(cntr.T)
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        marker='X',
        s=200,
        color='red',
        label='Centroids'
    )

    plt.title(f'Fuzzy C-Means Social Groups (K={final_k} | 2D PCA Projection)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('fuzzy_cmeans_scatter_plot.png')
    plt.close()

    print("--- 3. Cluster Visualization Complete (Plot Saved) ---")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        # 1. Preprocessing (Gender Removed)
        X_scaled, df_encoded, scaler = load_and_preprocess_data(FILE_NAME)

        # 2. Fuzzy C-Means Training
        cntr, u, fpc = train_fuzzy_c_means(X_scaled, FINAL_K, FUZZINESS_M)

        # 3. Evaluation, Visualization, and Saving
        df_final, hard_labels, final_centroids = evaluate_and_save_fuzzy_results(X_scaled, df_encoded, cntr, u, FINAL_K)
        visualize_fuzzy_clusters(X_scaled, hard_labels, FINAL_K, final_centroids)

        # 4. Final Centroid Interpretation (using membership weighted averages)
        # Note: Calculating fuzzy centroids explicitly is complex. We use skfuzzy's cntr output.
        print("\n--- Centroid Interpretation (FCM Centers) ---")
        centroid_df = pd.DataFrame(final_centroids.T, columns=df_encoded.columns)
        centroid_df.index = [f'Fuzzy_Group_{i}' for i in range(FINAL_K)]
        print(centroid_df)

    except ImportError:
        print("\n*** ERROR: scikit-fuzzy Library Missing ***")
        print("Please run '!pip install scikit-fuzzy' in a Colab cell and try again.")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print("An error occurred during data processing or FCM. Check file path/data integrity.")