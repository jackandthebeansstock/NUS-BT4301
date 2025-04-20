import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
import joblib
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

class LdaClustering:
    """
    A class to perform LDA-based clustering on user behavioural data.
    
    Parameters:
      features: list of feature names (default: ["diversity", "total_spent", "total_purchased"])
      alpha: float, Dirichlet prior for document-topic distribution (default: 1.0)
      beta: float, Dirichlet prior for topic-word distribution (default: 1.0)
      num_topics: int, number of topics (default: 4)
      max_iter: int, maximum number of iterations for LDA (default: 50)
      seed: int, random seed for reproducibility (default: 1)
      output_model_path: str, file path to save the LDA model (default: "lda_model.pkl")
      output_csv_path: str, file path to save the topic probabilities CSV (default: "user_topic_probabilities.csv")
      output_plot_pdf: str, file path to save the visualisation PDF (default: "lda_cluster_plots.pdf")
    """
    def __init__(self,
                 features=None,
                 alpha=1.0,
                 beta=1.0,
                 num_topics=4,
                 max_iter=50,
                 seed=1,
                 output_model_path="lda_model.pkl",
                 output_csv_path="user_topic_probabilities.csv",
                 output_plot_pdf="lda_cluster_plots.pdf"):
        if features is None:
            features = ["diversity", "total_spent", "total_purchased"]
        self.features = features
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.max_iter = max_iter
        self.seed = seed
        self.output_model_path = output_model_path
        self.output_csv_path = output_csv_path
        self.output_plot_pdf = output_plot_pdf
        self.lda = None
        self.data = None
        self.topic_distributions = None

    def _check_negative(self, X, df):
        negative_mask = (X < 0).any(axis=1)
        if negative_mask.any():
            invalid_rows = df.loc[negative_mask, ["user_id"] + self.features]
            print("Negative values found in the following rows:")
            print(invalid_rows.to_string(index=False))
            raise ValueError("Negative values detected in input features. Cannot proceed with LDA.")

    def fit(self, data: pd.DataFrame):
        """
        Fit the LDA model on the provided data.
        Exports topic probabilities to CSV, saves the model, and merges the results with the original data.
        """
        self.data = data.copy()
        user_ids = self.data["user_id"]
        X = self.data[self.features].values
        self._check_negative(X, self.data)
        
        self.lda = LDA(n_components=self.num_topics,
                       doc_topic_prior=self.alpha,
                       topic_word_prior=self.beta,
                       max_iter=self.max_iter,
                       random_state=self.seed)
        self.topic_distributions = self.lda.fit_transform(X)
        topic_columns = [f"topic_{i}" for i in range(self.num_topics)]
        df_probs = pd.DataFrame(self.topic_distributions, columns=topic_columns)
        df_probs.insert(0, "user_id", user_ids)
        df_probs["cluster"] = df_probs[topic_columns].values.argmax(axis=1)
        df_probs.to_csv(self.output_csv_path, index=False)
        joblib.dump(self.lda, self.output_model_path)
        self.data = self.data.merge(df_probs, on="user_id")
        print(f"Model saved to {self.output_model_path} and topic probabilities saved to {self.output_csv_path}")
        return self.data

    def visualize(self):
        """
        Generate a PDF containing various 2D and 3D visualisations of the clusters.
        """
        if self.data is None:
            raise ValueError("No data available. Fit the model before visualising.")
        with PdfPages(self.output_plot_pdf) as pdf_pages:
            def plot_scatter(df, x, y, title):
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=df[x], y=df[y], hue=df["cluster"], palette='tab10', alpha=0.6)
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title(title)
                plt.tight_layout()
                plt.grid(True)
                pdf_pages.savefig()
                plt.close()

            plot_scatter(self.data, 'diversity', 'total_spent', 'Diversity vs Total Spent')
            plot_scatter(self.data, 'total_spent', 'total_purchased', 'Total Spent vs Total Purchased')
            plot_scatter(self.data, 'total_purchased', 'diversity', 'Total Purchased vs Diversity')

            sample_df = self.data.sample(n=min(1000, len(self.data)), random_state=self.seed).copy()

            # Define CMYK reference for legend
            cmyk_reference = {
                "Cluster 0 (Cyan)":    (1, 0, 0, 0),
                "Cluster 1 (Magenta)": (0, 1, 0, 0),
                "Cluster 2 (Yellow)":  (0, 0, 1, 0),
                "Cluster 3 (Black)":   (0, 0, 0, 1),
            }

            def cmyk_to_rgb_single(c, m, y, k):
                r = 1.0 - min(1.0, c + k)
                g = 1.0 - min(1.0, m + k)
                b = 1.0 - min(1.0, y + k)
                return (r, g, b)

            def cmyk_to_rgb(cmyk_array):
                c, m, y, k = np.split(cmyk_array, 4, axis=1)
                r = 1.0 - np.minimum(1.0, c + k)
                g = 1.0 - np.minimum(1.0, m + k)
                b = 1.0 - np.minimum(1.0, y + k)
                return np.concatenate([r, g, b], axis=1)

            cmyk = sample_df[[f"topic_{i}" for i in range(self.num_topics)]].values
            cmyk = cmyk / cmyk.max()  # normalise
            rgb = np.clip(cmyk_to_rgb(cmyk), 0, 1)

            def plot_custom_3d(df, x, y, z, title):
                elev = 30
                azim = 45
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(df[x], df[y], df[z], c=rgb, alpha=0.7, s=8)
                ax.set_xlabel(x)
                ax.set_ylabel(y)
                ax.set_zlabel(z)
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

            plot_custom_3d(sample_df, "diversity", "total_spent", "total_purchased", "3D View (Total Purchased)")
            plot_custom_3d(sample_df, "total_spent", "total_purchased", "diversity", "3D View (Diversity)")
            plot_custom_3d(sample_df, "total_purchased", "diversity", "total_spent", "3D View (Total Spent)")

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

            plot_2d_projection(sample_df, "diversity", "total_spent", "2D Projection: Diversity vs Total Spent")
            plot_2d_projection(sample_df, "total_spent", "total_purchased", "2D Projection: Total Spent vs Total Purchased")
            plot_2d_projection(sample_df, "diversity", "total_purchased", "2D Projection: Diversity vs Total Purchased")
        
        print(f"Visualisation PDF saved to {self.output_plot_pdf}")
