import warnings
import subprocess
import sys
import os
import argparse
import json
import time
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    from transformers import pipeline
except ImportError:
    print("transformers not found. Installing it now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import pipeline
concepts = [
    "purchase", "intent", "shipping", "damage", "arrival", "product", "refund",
    "replacement", "dispute", "review", "condition", "packaging", "performance",
    "responsiveness", "characteristics", "trustworthiness", "communication",
    "delay", "loss", "quality", "fragility", "honesty", "deception",
    "efficiency", "confusion", "clarity", "helpfulness", "negligence",
    "availability", "absence", "satisfaction", "dissatisfaction",
    "experience", "value", "service", "support", "interaction", "feedback",
    "durability", "design", "authenticity", "expectation", "engagement",
    "courtesy", "attitude", "loyalty", "reliability", "simplicity",
    "complexity", "information", "interface"
]
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_review(review_text: str, top_n: int = 3) -> dict:
    """
    Classify a review using zero-shot learning.
    Args:
        review_text (str): Text to classify.
        top_n (int): Number of top concepts to return.
    Returns:
        dict: Results with time and top concepts.
    """
    start = time.time()
    result = classifier(review_text, concepts, multi_label=True)
    ranked = sorted(zip(result['labels'], result['scores']), key=lambda x: x[1], reverse=True)
    elapsed = time.time() - start

    return {
        "time_taken_seconds": round(elapsed, 3),
        "top_concepts": [
            {"label": label, "score": round(score, 4)}
            for label, score in ranked[:top_n]
        ]
    }

def write_to_csv(json_result: dict, output_file: str) -> None:
    """
    Write classification results to a CSV file.
    """
    import csv
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["label", "score"])
        writer.writeheader()
        for row in json_result["top_concepts"]:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Zero-shot classify a review.")
    parser.add_argument("text", type=str, help="Review text to classify.")
    parser.add_argument("--top_n", type=int, default=3, help="Top N concepts to return.")
    parser.add_argument("--csv", type=str, help="Optionally write results to CSV file.")
    args = parser.parse_args()

    try:
        output = classify_review(args.text, top_n=args.top_n)
        print(json.dumps(output, indent=2))

        if args.csv:
            write_to_csv(output, args.csv)
            print(f"\nResults written to: {args.csv}")

    except Exception as e:
        print(f"Error during classification: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

# examples
# python classify.py "Very helpful and kind support team!" --top_n 5