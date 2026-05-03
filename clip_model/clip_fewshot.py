import torch
import clip
import os
from PIL import Image

class CLIPFewShotClassifier:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.prototypes = {}

    #######################################################################################
    def build_support(self, support_dir: str, save_path: str = "embeddings/clip_support.pt"):
        for label in os.listdir(support_dir):
            class_path = os.path.join(support_dir, label)

            if not os.path.isdir(class_path):
                continue


            embeddings = []
            for fname in os.listdir(class_path):
                fpath = os.path.join(class_path, fname)
                
                try:
                    image = self.preprocess(Image.open(fpath)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        emb = self.model.encode_image(image).squeeze()
                        emb /= emb.norm()

                    embeddings.append(emb)
                except Exception as e:
                    print(f"Skipping {fname}: {e}")


            if embeddings: 
                prototype = torch.stack(embeddings).mean(dim=0)
                prototype /= prototype.norm()

                self.prototypes[label] = prototype
                print(f"    {label}: {len(embeddings)} Image encoded")


        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.prototypes, save_path)
        print(f"Saved CLIP Few-shot Prototypes at: {save_path}")

    #######################################################################################
    def load_support(self, save_path: str = "embeddings/clip_support.pt"):
        self.prototypes = torch.load(save_path, map_location=self.device)

    #######################################################################################
    def predict(self, image_path: str) -> tuple[str, dict]:
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_emb = self.model.encode_image(image).squeeze()
            img_emb /= img_emb.norm()

        scores = {
            label: (proto @ img_emb).item()
            for label, proto in self.prototypes.items()
        }

        predicted_label = max(scores, key=scores.get)
        
        return predicted_label, scores