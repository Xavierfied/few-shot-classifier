from utils.args import get_res_args
from utils.helpers import create_runner, save_results

def main():
    args = get_res_args()

    print(f"\nModel: {args.model}")
    print(f"Source: {args.source}")

    # Load and Running
    classifier = create_runner(args.model, args)

    label, scores = classifier.predict(args.source)

    # Print Results
    print(f"Predicted: {label}")
    print("Scores:")

    for room, score_value in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    {room}: {score_value:.4f}")

    # Save Result:
    save_results(label, scores, args.source, args.output_dir)


if __name__ == "__main__":
    main()