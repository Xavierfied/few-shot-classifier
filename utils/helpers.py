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
        from clip_model.clip_zeroshot import CLIPNoShotClassifier
        
        classifier = CLIPNoShotClassifier(args.labels)
        return classifier
        


    elif model == "clip_fewshot":
        from clip_model.clip_fewshot import CLIPFewShotClassifier

        classifier = CLIPFewShotClassifier()
        support_path = args.embeddings
        if support_path == "embeddings/support.pt":
            support_path = "embeddings/clip_support.pt"

        classifier.load_support(support_path)
        return classifier

    else:
        raise ValueError(f"Unknown model: {model}")
    

###########################################################################
def save_results(label:str, scores: dict, source: str, output_dir: str, model: str = None):
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
    if model:
        fname = f"{stem}_{model}_result.json"
    else:
        fname = f"{stem}_result.json"

    out_path = Path(output_dir) / fname

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved at: {out_path}")
    return out_path