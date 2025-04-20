"""
Performs GMM-based clustering on user behavioural data using pandas and scikit-learn.
Now extended to:
1. Find the best number of components via BIC.
2. Plot cluster ellipses showing overlaps.
3. Increment clusters by +1 to label clusters 1..4 (C, M, Y, K).
4. Provide a pie chart of dominant clusters in CMYK colors.
5. Wrap the logic in a GmmClustering class with methods to load/save a model,
   assign clusters to new users, store user data, and produce visualisations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse, Patch
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import joblib

class GmmClustering:
    """
    A class to perform Gaussian Mixture Model clustering on user behavioural data.

    Attributes:
    -----------
    scaler : StandardScaler
        The fitted scaler for numeric features.
    gmm : GaussianMixture
        The trained Gaussian Mixture Model.
    features : list
        List of feature column names, e.g. ["diversity", "total_spent", "total_purchased"].
    random_state : int
        Random seed for reproducibility.
    pdf_path : str
        Path to save the PDF with visualisations.
    data : pd.DataFrame
        The main user DataFrame. After fitting, includes cluster probabilities and dominant clusters.

    Methods:
    --------
    find_best_n_components(X, min_comp=1, max_comp=8):
        Find the best number of components by BIC.
    fit(data: pd.DataFrame):
        Fit the GMM to the data, scaling features, generating probabilities, and storing clusters.
    save_model(path: str):
        Save the trained GMM model to disk.
    load_model(path: str):
        Load a previously trained GMM model from disk.
    assign_cluster(new_user: dict) -> int:
        Assign a single new user to a dominant cluster. Returns that cluster number.
    assign_clusters(new_users: pd.DataFrame) -> pd.DataFrame:
        Assign a list of new, unseen users to clusters. Returns a DataFrame with cluster info.
    store(new_user: dict):
        Append a single user's data to the main DataFrame, optionally after assigning cluster.
    visualize_clusters():
        Create a PDF of 2D, 3D, ellipse overlap, and pie chart visuals for the current data.
    """

    def __init__(self,
                 features=None,
                 pdf_path="gmm_cluster_plots_all.pdf",
                 random_state=0):
        """
        Initialize the GmmClustering class with default or user-defined features,
        a PDF path, and random_state.

        Parameters
        ----------
        features : list, optional
            The features used for clustering (default: ["diversity", "total_spent", "total_purchased"]).
        pdf_path : str, optional
            File path to save the PDF outputs (default: "gmm_cluster_plots.pdf").
        random_state : int, optional
            The random seed for reproducibility (default: 0).
        """
        if features is None:
            features = ["diversity", "total_spent", "total_purchased"]
        self.features = features
        self.pdf_path = pdf_path
        self.random_state = random_state
        self.scaler = None
        self.gmm = None
        self.data = None

    def find_best_n_components(self, X, min_comp=1, max_comp=8):
        """
        Find the best number of components for GMM by fitting each possible number
        from min_comp to max_comp and comparing BIC.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (already scaled if relevant).
        min_comp : int, optional
            Minimum number of components to consider (default: 1).
        max_comp : int, optional
            Maximum number of components to consider (default: 8).

        Returns
        -------
        int
            The optimal number of components based on the lowest BIC.
        """
        best_bic = float("inf")
        best_n = min_comp
        for n in range(min_comp, max_comp+1):
            gmm_temp = GaussianMixture(n_components=n, random_state=self.random_state)
            gmm_temp.fit(X)
            bic = gmm_temp.bic(X)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        print(f"Best number of components by BIC: {best_n} (BIC={best_bic:.2f})")
        return best_n

    def fit(self, data: pd.DataFrame, n_components=4, max_iter=1000):
        """
        Fit the GMM on the given data, scaling features, generating cluster probabilities,
        and storing results in self.data.

        Parameters
        ----------
        data : pd.DataFrame
            The user DataFrame containing the features.
        n_components : int, optional
            Number of mixture components (default: 4).
        max_iter : int, optional
            Maximum EM iterations (default: 1000).
        """
        self.data = data.copy()
        X_raw = self.data[self.features].values

        # Check for negative values
        negative_mask = (X_raw < 0).any(axis=1)
        if negative_mask.any():
            invalid_rows = self.data.loc[negative_mask, ["user_id"] + self.features]
            print("Negative values found in the following rows:")
            print(invalid_rows.to_string(index=False))
            raise ValueError("Negative values detected in input features. Cannot proceed with GMM.")

        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        # Fit GMM
        self.gmm = GaussianMixture(n_components=n_components,
                                   max_iter=max_iter,
                                   random_state=self.random_state)
        self.gmm.fit(X_scaled)

        # Generate probabilities
        probs = self.gmm.predict_proba(X_scaled)

        # Argmax for dominant cluster, add +1 to shift from 0..3 => 1..4
        labels = np.argmax(probs, axis=1) + 1

        # Create DataFrame of cluster probabilities
        cluster_columns = [f"cluster_{i}" for i in range(1, n_components+1)]
        df_probs = pd.DataFrame(probs, columns=cluster_columns)
        df_probs["dominant_cluster"] = labels

        # Merge with original
        self.data = pd.concat([self.data.reset_index(drop=True), df_probs], axis=1)

    def save_model(self, path: str = "gmm_model_all.pkl"):
        """
        Save the fitted GMM model to disk.

        Parameters
        ----------
        path : str, optional
            File path to save the GMM model (default: "gmm_model.pkl").
        """
        if not self.gmm:
            raise ValueError("No model found. Fit the model first before saving.")
        joblib.dump((self.gmm, self.scaler, self.features), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = "gmm_model_all.pkl", data: pd.DataFrame = None):
        """
        Load a previously trained GMM model from disk. Optionally set the data attribute.

        Parameters
        ----------
        path : str, optional
            Path where the model is saved (default: "gmm_model.pkl").
        data : pd.DataFrame, optional
            If provided, sets self.data to this DataFrame (default: None).
        """
        self.gmm, self.scaler, self.features = joblib.load(path)
        if data is not None:
            self.data = data.copy()
        print(f"Model loaded from {path}")

    def assign_cluster(self, new_user: dict) -> int:
        """
        Assign a single new user (dictionary of feature values) to a dominant cluster.

        Parameters
        ----------
        new_user : dict
            E.g. {"user_id": "ABC123", "diversity": 5, "total_spent": 12.5, "total_purchased": 4}

        Returns
        -------
        int
            The assigned cluster number (1..N).
        """
        if not self.gmm or not self.scaler:
            raise ValueError("Model not loaded or fitted. Call load_model or fit first.")

        # Convert dict to array in correct feature order
        row = np.array([[new_user[feat] for feat in self.features]], dtype=float)
        row_scaled = self.scaler.transform(row)
        probs = self.gmm.predict_proba(row_scaled)
        cluster = np.argmax(probs, axis=1)[0] + 1  # shift +1
        return cluster

    def assign_clusters(self, new_users: pd.DataFrame) -> pd.DataFrame:
        """
        Assign a list of new, unseen users to clusters. Returns their cluster membership.

        Parameters
        ----------
        new_users : pd.DataFrame
            A DataFrame with columns matching self.features.

        Returns
        -------
        pd.DataFrame
            DataFrame with cluster probabilities and a 'dominant_cluster' column.
        """
        if not self.gmm or not self.scaler:
            raise ValueError("Model not loaded or fitted. Call load_model or fit first.")

        X_raw = new_users[self.features].values
        X_scaled = self.scaler.transform(X_raw)
        probs = self.gmm.predict_proba(X_scaled)
        labels = np.argmax(probs, axis=1) + 1

        cluster_cols = [f"cluster_{i}" for i in range(1, self.gmm.n_components+1)]
        df_probs = pd.DataFrame(probs, columns=cluster_cols)
        df_probs["dominant_cluster"] = labels
        # Optionally merge with user IDs if present
        if "user_id" in new_users.columns:
            df_probs.insert(0, "user_id", new_users["user_id"].values)

        return df_probs

    def store(self, new_user: dict):
        """
        Store a single new user's data in self.data. Also assigns cluster if model is present.

        Parameters
        ----------
        new_user : dict
            E.g. {"user_id": "ABC123", "diversity": 5, "total_spent": 12.5, "total_purchased": 4}
        """
        new_row = {k: new_user[k] for k in self.features if k in new_user}
        # If model is loaded, assign cluster
        if self.gmm:
            cluster = self.assign_cluster(new_user)
            # Also get full proba
            row = np.array([[new_user[feat] for feat in self.features]], dtype=float)
            row_scaled = self.scaler.transform(row)
            probs = self.gmm.predict_proba(row_scaled)[0]
            # Build the row
            for i, prob in enumerate(probs, start=1):
                new_row[f"cluster_{i}"] = prob
            new_row["dominant_cluster"] = cluster
        if "user_id" in new_user:
            new_row["user_id"] = new_user["user_id"]
        # Append
        self.data = pd.concat([self.data,pd.Series(new_row)], ignore_index = True)

    def visualize_clusters(self):
        """
        Return a PDF with:
         1) 2D scatter plots (RGB from CMYK blending).
         2) 3D scatter plots.
         3) Ellipse overlap plots for each pair of features.
         4) A pie chart of cluster distribution.

        The PDF is saved to self.pdf_path.
        """
        if self.data is None or "cluster_1" not in self.data.columns:
            raise ValueError("No data with cluster probabilities. Please call fit first.")

        # Sample for plotting
        sample_df = self.data.sample(n=min(1000000, len(self.data)), random_state=self.random_state).copy()

        # Build the CMYK array and convert to RGB
        cluster_cols = [f"cluster_{i}" for i in range(1, self.gmm.n_components+1)]
        cmyk_array = sample_df[cluster_cols].values
        cmyk_array /= cmyk_array.max()  # normalise
        def cmyk_to_rgb(cmyk_array):
            c, m, y, k = np.split(cmyk_array, 4, axis=1)
            r = 1.0 - np.minimum(1.0, c + k)
            g = 1.0 - np.minimum(1.0, m + k)
            b = 1.0 - np.minimum(1.0, y + k)
            return np.concatenate([r, g, b], axis=1)
        rgb = np.clip(cmyk_to_rgb(cmyk_array), 0, 1)

        # CMYK reference for legend
        cmyk_reference = {
            "Cluster 1 (Cyan)":    (1, 0, 0, 0),
            "Cluster 2 (Magenta)": (0, 1, 0, 0),
            "Cluster 3 (Yellow)":  (0, 0, 1, 0),
            "Cluster 4 (Black)":   (0, 0, 0, 1),
        }
        def cmyk_to_rgb_single(c, m, y, k):
            r = 1.0 - min(1.0, c + k)
            g = 1.0 - min(1.0, m + k)
            b = 1.0 - min(1.0, y + k)
            return (r, g, b)

        # Start PDF
        with PdfPages(self.pdf_path) as pdf_pages:

            # ---------- 3D plots ----------
            def plot_custom_3d(df, elev, azim, title):
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(df["diversity"], df["total_spent"], df["total_purchased"],
                           c=rgb, alpha=0.7, s=8)
                ax.set_xlabel("Diversity")
                ax.set_ylabel("Total Spent")
                ax.set_zlabel("Total Purchased")
                ax.set_box_aspect([1, 1, 1])
                ax.view_init(elev=elev, azim=azim)
                ax.set_title(title)
                legend_handles = [
                    Patch(color=cmyk_to_rgb_single(*cmyk), label=label)
                    for label, cmyk in cmyk_reference.items()
                ]
                plt.legend(handles=legend_handles, title="Dominant Cluster (CMYK)")
                pdf_pages.savefig()
                plt.close()

            plot_custom_3d(sample_df, 30, 45, "3D View (Default Angle)")
            plot_custom_3d(sample_df, 30, 90, "3D View (Total Spent as Height)")
            plot_custom_3d(sample_df, 60, 0, "3D View (Diversity as Height)")

            # ---------- 2D scatter plots ----------
            def plot_2d_projection(df, x, y, title):
                plt.figure(figsize=(8, 6))
                plt.scatter(df[x], df[y], c=rgb, alpha=0.7, s=8)
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title(title)
                plt.tight_layout()
                plt.grid(True)
                legend_handles = [
                    Patch(color=cmyk_to_rgb_single(*cmyk), label=label)
                    for label, cmyk in cmyk_reference.items()
                ]
                plt.legend(handles=legend_handles, title="Dominant Cluster (CMYK)")
                pdf_pages.savefig()
                plt.close()

            plot_2d_projection(sample_df, "diversity", "total_spent", "2D: Diversity vs Total Spent")
            plot_2d_projection(sample_df, "total_spent", "total_purchased", "2D: Total Spent vs Total Purchased")
            plot_2d_projection(sample_df, "diversity", "total_purchased", "2D: Diversity vs Total Purchased")

            # ---------- Ellipse overlap plots ----------
            # We'll do pairwise 2D ellipse for each pair of features
            def plot_cluster_ellipses(df, x, y, title):
                """
                Plot 2D data plus GMM cluster ellipses for each cluster.
                We'll color each cluster center with the base CMYK color.
                """
                scaler_temp = StandardScaler().fit(df[[x, y]])
                X_for_ellipse = scaler_temp.transform(df[[x, y]])
                labels_for_ellipse = df["dominant_cluster"].values
                plt.figure(figsize=(8, 6))
                plt.scatter(X_for_ellipse[:, 0], X_for_ellipse[:, 1], c=labels_for_ellipse, cmap='tab10', alpha=0.5, s=8)
                plt.title(title)
                plt.xlabel(x)
                plt.ylabel(y)

                # Plot ellipses for each cluster
                means = self.gmm.means_[:, [self.features.index(x), self.features.index(y)]]
                covariances = self.gmm.covariances_
                # Cov might be full, so pick the sub-cov
                for i in range(self.gmm.n_components):
                    subcov = covariances[i][[self.features.index(x), self.features.index(y)]][:, [self.features.index(x), self.features.index(y)]]
                    mean_2d_2d = means[i]
                    mean_2d_2d_df = pd.DataFrame([mean_2d_2d], columns=[x, y])
                    mean_2d = scaler_temp.transform(mean_2d_2d_df)[0]

                    # Compute ellipse from subcov
                    vals, vecs = np.linalg.eigh(subcov)
                    # scale by the scaler
                    vals = np.abs(vals)
                    order = vals.argsort()[::-1]
                    vals, vecs = vals[order], vecs[:, order]

                    angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                    width, height = 2 * 2.0 * np.sqrt(vals)  # n-std ~2
                    e = Ellipse(xy=(mean_2d[0], mean_2d[1]), width=width, height=height,
                            angle=angle, edgecolor='red', fill=False, lw=2)
                    plt.gca().add_patch(e)

                pdf_pages.savefig()
                plt.close()

            # For each pair of features
            pairs = [("diversity", "total_spent"), ("total_spent", "total_purchased"), ("diversity", "total_purchased")]
            for (a, b) in pairs:
                plot_cluster_ellipses(sample_df, a, b, f"Cluster Overlaps: {a} vs {b}")

            # ---------- Pie chart of dominant clusters ----------
            cluster_counts = self.data["dominant_cluster"].value_counts().sort_index()
            # For CMYK: 1->C, 2->M, 3->Y, 4->K
            cmyk_colors = ["cyan", "magenta", "yellow", "lightgrey"]
            plt.figure(figsize=(5, 5))
            plt.pie(cluster_counts.values, labels=[f"Cluster {i}" for i in cluster_counts.index],
                    colors=[cmyk_colors[i-1] for i in cluster_counts.index],
                    autopct="%1.1f%%", startangle=140)
            plt.title("Dominant Cluster Distribution")
            pdf_pages.savefig()
            plt.close()

        print(f"Visualisation PDF saved to {self.pdf_path}")


# -------------- Usage Example --------------
if __name__ == "__main__":
    # Suppose we have data in user_category_metrics.csv
    # We'll load it here in code to demonstrate usage:
    data_path = "user_category_metrics.csv"
    df_main = pd.read_csv(data_path)

    # 1) Instantiate the class
    gmm_clustering = GmmClustering(random_state=0)

    # 2) OPTIONAL: find best n_components via BIC
    # We'll do a quick example:
    # best_n = gmm_clustering.find_best_n_components(df_main[["diversity", "total_spent", "total_purchased"]].values, 1, 8)

    # 3) Fit with a chosen number of components
    gmm_clustering.fit(df_main, n_components=4, max_iter=1000)

    # 4) Save the model
    gmm_clustering.save_model("gmm_model.pkl")

    # 5) Visualise clusters
    gmm_clustering.visualize_clusters()

    # 6) Assign cluster to a new user
    new_user_dict = {"user_id": "AAA123", "diversity": 5, "total_spent": 100.0, "total_purchased": 3}
    assigned_cluster = gmm_clustering.assign_cluster(new_user_dict)
    print(f"New user assigned to cluster {assigned_cluster}")

    # 7) Assign clusters to a batch of new users
    new_users_df = pd.DataFrame([
        {"user_id": "BBB222", "diversity": 1, "total_spent": 5.0, "total_purchased": 1},
        {"user_id": "CCC333", "diversity": 7, "total_spent": 200.0, "total_purchased": 10}
    ])
    assigned_df = gmm_clustering.assign_clusters(new_users_df)
    print("Batch new users assigned:\n", assigned_df)

    # 8) Store a new user in the main data
    gmm_clustering.store(new_user_dict)
