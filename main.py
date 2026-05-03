from utils.args import get_res_args
from utils.helpers import create_runner, save_results
import json
import time
from pathlib import Path
import os

def main():
    args = get_res_args()

    print(f"\nModel: {args.model}")
    print(f"Source: {args.source}")

    # If compare_all is requested, run every model and save one combined JSON
    if getattr(args, "compare_all", False):
        models = ["resnet_cosine", "clip_zeroshot", "clip_fewshot"]
        combined = {}

        for m in models:
            print(f"\nRunning: {m}")
            clf = create_runner(m, args)
            pred, scores = clf.predict(args.source)
            # keep scores ordered by confidence
            ordered = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
            combined[m] = {"predicted": pred, "scores": ordered}

        # Save combined result
        os.makedirs(args.output_dir, exist_ok=True)
        stem = Path(args.source).stem
        out_path = Path(args.output_dir) / f"{stem}_comparison_result.json"
        payload = {
            "source": args.source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": combined
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Comparison saved at: {out_path}")
        return

    # Load and Running (single model)
    classifier = create_runner(args.model, args)

    label, scores = classifier.predict(args.source)

    # Print Results
    print(f"Predicted: {label}")
    print("Scores:")

    for room, score_value in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    {room}: {score_value:.4f}")

    # Save Result:
    save_results(label, scores, args.source, args.output_dir, args.model)


if __name__ == "__main__":
    main()