import argparse
import pandas as pd
from lda import LdaClustering

def main():
    parser = argparse.ArgumentParser(description="LDA Clustering CLI interface")
    parser.add_argument("data", type=str, help="Path to the input CSV data file")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Dirichlet prior for topic mixture (default: 1.0)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Dirichlet prior for topic-word distribution (default: 1.0)")
    parser.add_argument("--num_topics", type=int, default=4,
                        help="Number of topics (default: 4)")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum iterations for LDA (default: 50)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reproducibility (default: 1)")
    parser.add_argument("--output_model", type=str, default="lda_model.pkl",
                        help="Path to save the LDA model (default: lda_model.pkl)")
    parser.add_argument("--output_csv", type=str, default="user_topic_probabilities.csv",
                        help="Path to save topic probabilities CSV (default: user_topic_probabilities.csv)")
    parser.add_argument("--output_pdf", type=str, default="lda_cluster_plots.pdf",
                        help="Path to save the visualisation PDF (default: lda_cluster_plots.pdf)")
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    lda_cluster = LdaClustering(alpha=args.alpha,
                                beta=args.beta,
                                num_topics=args.num_topics,
                                max_iter=args.max_iter,
                                seed=args.seed,
                                output_model_path=args.output_model,
                                output_csv_path=args.output_csv,
                                output_plot_pdf=args.output_pdf)
    lda_cluster.fit(df)
    lda_cluster.visualize()
    print("LDA clustering completed.")
    print(f"Model saved to: {args.output_model}")
    print(f"Topic probabilities saved to: {args.output_csv}")
    print(f"Visualisation PDF saved to: {args.output_pdf}")

if __name__ == "__main__":
    main()
