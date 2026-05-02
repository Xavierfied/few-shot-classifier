import os
from pathlib import Path
import json
import time


###########################################################################
def create_runner(model:str, args):
    """
    calls the appropriate classifier as required based on "--model"
    """
    if model == "resnet_cosine":
        from resnet_fewshot.infer import RoomClassifier
        return RoomClassifier(embeddings_path=args.embeddings)
    
    elif model == "clip_zeroshot":
        pass

    elif model == "clip_fewshot":
        pass

    else:
        raise ValueError(f"Unknown model: {model}")
    

###########################################################################
def save_results(label:str, scores: dict, source: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Keep highest-confidence class at the top in saved output.
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    result = {
        "source": source,
        "predicted": label,
        "scores": sorted_scores,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    stem = Path(source).stem
    out_path = Path(output_dir) / f"{stem}_result.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved at: {out_path}")
    return out_path