import argparse
import pandas as pd
import json
from gmm import GmmClustering

def main():
    parser = argparse.ArgumentParser(description="GMM clustering CLI interface")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Fit a GMM to user data")
    fit_parser.add_argument("csv", type=str, help="Path to input CSV")
    fit_parser.add_argument("--n_components", type=int, default=4, help="Number of GMM components")
    fit_parser.add_argument("--save_model", type=str, default="gmm_model.pkl", help="Where to save the model")

    # Assign cluster (single user)
    assign_parser = subparsers.add_parser("assign", help="Assign a new user to a cluster")
    assign_parser.add_argument("model", type=str, help="Path to saved model")
    assign_parser.add_argument("json", type=str, help="Path to JSON with new user data")

    # Assign clusters (batch)
    assign_batch_parser = subparsers.add_parser("assign_batch", help="Assign clusters to a batch of users")
    assign_batch_parser.add_argument("model", type=str, help="Path to saved model")
    assign_batch_parser.add_argument("csv", type=str, help="Path to CSV of new users")

    # Visualisation
    vis_parser = subparsers.add_parser("visualise", help="Generate clustering visualisation PDF")
    vis_parser.add_argument("model", type=str, help="Path to saved model")
    vis_parser.add_argument("csv", type=str, help="Path to CSV of clustered data")

    args = parser.parse_args()

    if args.command == "fit":
        df = pd.read_csv(args.csv)
        gmm = GmmClustering()
        gmm.fit(df, n_components=args.n_components)
        gmm.save_model(args.save_model)
        print(f"Model trained and saved to {args.save_model}")

    elif args.command == "assign":
        user_data = json.load(open(args.json))
        gmm = GmmClustering()
        gmm.load_model(args.model)
        cluster = gmm.assign_cluster(user_data)
        print(json.dumps({"assigned_cluster": cluster}, indent=2))

    elif args.command == "assign_batch":
        df = pd.read_csv(args.csv)
        gmm = GmmClustering()
        gmm.load_model(args.model)
        result_df = gmm.assign_clusters(df)
        print(result_df.to_string(index=False))

    elif args.command == "visualise":
        df = pd.read_csv(args.csv)
        gmm = GmmClustering()
        gmm.load_model(args.model, data=df)
        gmm.visualize_clusters()

if __name__ == "__main__":
    main()
