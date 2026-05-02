import argparse

"""
ResNet50 Args
"""

def get_res_args():
    parser = argparse.ArgumentParser(description="ResNet50 Few Shotting Tool")

    parser.add_argument("--source",
                        type=str,
                        required=True,
                        help="Enter the Image's path you want to identify.")
    
    parser.add_argument("--output_dir",
                        type=str,
                        default="results",
                        help="Dir to save results in.")
    
    parser.add_argument("--model", required=True,
                        choices=["resnet_cosine", "clip_zeroshot", "clip_fewshot"],
                        help="Which model to run.")
    
    parser.add_argument("--embeddings", type=str, default="embeddings/support.pt",
                        help="Path to precomputed embeddings (ResNet / CLIP few-shot).")
    
    parser.add_argument("--labels", type=str, nargs="+",
                        default=["kitchen", "bedroom", "bathroom", "living room"],
                        help="Labels for CLIP zero-shot. e.g. --labels kitchen bedroom")

    return parser.parse_args()