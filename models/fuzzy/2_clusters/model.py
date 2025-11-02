import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import skfuzzy as fuzz # NOTE: Requires installation: !pip install scikit-fuzzy
from sklearn.metrics import silhouette_score

# --- FILE AND MODEL CONFIGURATION ---
FILE_NAME = 'featureEngineeredBetter.csv'
FINAL_OUTPUT_CSV = 'social_clustered_users_fuzzy.csv'
# FINAL_K is no longer a hardcoded input, but the search range max
MAX_K_SEARCH = 10
FUZZINESS_M = 2.0 # Fuzziness parameter (m > 1). Higher m means fuzzier clusters.

# --- 1. MODULARIZED FUNCTIONS ---

def load_and_preprocess_data(file_path):
    """Loads data, handles cleaning, imputation, encoding, and scaling."""
    print("--- 1. Data Preprocessing Started ---")

    # Use header=1 to correctly read the CSV, skipping the first row
    try:
        df = pd.read_csv(file_path, encoding='unicode_escape', header=1)
    except pd.errors.ParserError:
        # Fallback if header=1 causes issues or if the file structure is different
        df = pd.read_csv(file_path, encoding='unicode_escape')

    # Data Cleaning: Convert byte-strings to standard strings
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].astype(str).str.startswith("b'").any():
            df[col] = df[col].astype(str).str.replace("b'", "", regex=False).str.replace("'", "", regex=False)

    # Define features for clustering (Assuming 'Gender' or other non-features are removed/excluded if necessary)
    # Automatically identify categorical and numerical columns after initial load
    df_numeric = df.select_dtypes(include=np.number)
    df_object = df.select_dtypes(include='object')

    numerical_cols = df_numeric.columns.tolist()
    categorical_cols = df_object.columns.tolist()

    # a. Handle Missing Values (Imputation)
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # b. One-Hot Encoding for categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure all remaining columns are numerical before scaling
    X = df_encoded.values.astype(float)

    # Feature Scaling: Standardize all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Features used for clustering: {X_scaled.shape[1]}")
    print("--- 1. Data Preprocessing Complete ---")
    return X_scaled, df_encoded, scaler

# NEW FUNCTION TO FIND OPTIMAL K
def find_optimal_k_fcm(X_scaled, max_k, fuzziness_m):
    """
    Finds the optimal number of clusters K using the Fuzzy Partition Coefficient (FPC) and Partition Entropy (PE).
    K that MAXIMIZES FPC is the preferred choice.
    """
    print(f"\n--- Searching for Optimal K (2 to {max_k}) ---")
    fpc_values = []
    pe_values = []
    k_range = range(2, max_k + 1)

    # Transpose X_scaled for skfuzzy (N_features x N_samples)
    X_fcm = X_scaled.T

    for k in k_range:
        # Train FCM for current K
        try:
            # skfuzzy.cmeans returns [cntr, u, u0, d, jm, p, fpc]
            _, _, _, _, _, p, fpc = fuzz.cluster.cmeans(
                X_fcm, c=k, m=fuzziness_m, error=0.005, maxiter=1000, seed=42
            )
            fpc_values.append(fpc)
            pe_values.append(p)
            print(f"K={k}: FPC={fpc:.4f}, PE={p:.4f}")
        except ValueError as e:
            # Stop if k is too high for the dataset or other issues
            print(f"Stopping K search at {k} due to error: {e}")
            break

    if not fpc_values:
        print("No valid K found. Falling back to K=5.")
        return 5

    # Optimal K is the one with maximum FPC
    k_candidates = k_range[:len(fpc_values)]
    optimal_k_index = np.argmax(fpc_values)
    chosen_k = k_candidates[optimal_k_index]

    optimal_k_pe = k_candidates[np.argmin(pe_values)] # For comparison

    print(f"\nOptimal K (Max FPC): {chosen_k}")
    print(f"Optimal K (Min PE): {optimal_k_pe}")
    print(f"Chosen K for Final Model: {chosen_k}")

    # Plotting for visualization of K selection
    plt.figure(figsize=(10, 5))
    plt.plot(k_candidates, fpc_values, 'bo-', label='Fuzzy Partition Coefficient (FPC) - Maximize')
    # Plot 1 - PE since PE minimizes (easier visualization)
    plt.plot(k_candidates, [1 - val for val in pe_values], 'rx--', label='1 - Partition Entropy (PE) - Maximize')
    plt.axvline(x=chosen_k, color='g', linestyle='--', label=f'Chosen K={chosen_k}')
    plt.title('Fuzzy Clustering Validity Metrics vs. Number of Clusters (K)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('optimal_k_fcm_metrics.png')
    plt.close()

    return chosen_k


def train_fuzzy_c_means(X_scaled, final_k, fuzziness_m):
    """Trains the Fuzzy C-Means model and returns centroids and membership matrix."""
    print(f"--- 2. Fuzzy C-Means Training Started (K={final_k}, m={fuzziness_m}) ---")

    # Transpose X_scaled  because skfuzzy expects features on the rows (N_features x N_samples)
    X_fcm = X_scaled.T

    # Train FCM: The output `u` is the membership matrix, `cntr` is the cluster centers
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
    scatter = plt.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        c=pca_df['Hard_Label'],
        cmap='tab10',
        marker='o',
        s=50,
        alpha=0.6,
        label=None # Do not label the scatter plot here
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
        edgecolors='black',
        linewidths=1.5,
        label='Centroids'
    )

    plt.title(f'Fuzzy C-Means Social Groups (K={final_k} | 2D PCA Projection)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')

    # Create legend handles for the clusters
    legend1 = plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.gca().add_artist(legend1)

    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig('fuzzy_cmeans_scatter_plot.png')
    plt.close()

    print("--- 3. Cluster Visualization Complete (Plot Saved) ---")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        # 1. Preprocessing
        X_scaled, df_encoded, scaler = load_and_preprocess_data(FILE_NAME)

        # 2. Determine Optimal K
        optimal_k = find_optimal_k_fcm(X_scaled, max_k=MAX_K_SEARCH, fuzziness_m=FUZZINESS_M)

        # 3. Fuzzy C-Means Training with Optimal K
        cntr, u, fpc = train_fuzzy_c_means(X_scaled, optimal_k, FUZZINESS_M)

        # 4. Evaluation, Visualization, and Saving
        df_final, hard_labels, final_centroids = evaluate_and_save_fuzzy_results(X_scaled, df_encoded, cntr, u, optimal_k)
        visualize_fuzzy_clusters(X_scaled, hard_labels, optimal_k, final_centroids)

        # 5. Final Centroid Interpretation
        print("\n--- Centroid Interpretation (FCM Centers) ---")

        # Centroids are in the scaled feature space (cntr.T)
        final_centroids_T = final_centroids.T

        # Inverse transform to get back to original data scale for interpretability
        centroid_original_scale = scaler.inverse_transform(final_centroids_T)

        # Create a DataFrame for scaled centroids
        centroid_scaled_df = pd.DataFrame(final_centroids_T, columns=df_encoded.columns)
        centroid_scaled_df.index = [f'Fuzzy_Group_{i}' for i in range(optimal_k)]
        print("\n**Scaled Centroids (Mean of each feature in the cluster):**")
        print(centroid_scaled_df)

        # Create a DataFrame for original scale centroids (for business interpretation)
        centroid_original_df = pd.DataFrame(centroid_original_scale, columns=df_encoded.columns)
        centroid_original_df.index = [f'Fuzzy_Group_{i}' for i in range(optimal_k)]
        print("\n**Original Scale Centroids (Interpretable Cluster Means):**")
        print(centroid_original_df.round(2))

    except ImportError:
        print("\n*** ERROR: scikit-fuzzy Library Missing ***")
        print("The optimal K search and FCM model require 'scikit-fuzzy'.")
        print("Please run '!pip install scikit-fuzzy' in a Colab cell and try again.")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        print("An error occurred during data processing or FCM. Check file path/data integrity.")