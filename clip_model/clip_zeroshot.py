import torch
import clip
from PIL import Image


class CLIPNoShotClassifier:
    def __init__(self, labels: list[str], model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.labels = labels

        text_inputs = clip.tokenize([f"{label}" for label in labels])
        with torch.no_grad():
            self.text_embeddings = self.model.encode_text(text_inputs.to(self.device))
            self.text_embeddings /= self.text_embeddings.norm(dim=1, keepdim=True)

    
    def predict(self, image_path:str) -> tuple[str, dict]:
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_emb = self.model.encode_image(image)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)

        similarities = (self.text_embeddings @ img_emb.T).flatten().tolist()
        
        scores = dict(zip(self.labels, similarities))
        
        predicted_label = max(scores, key=scores.get)
        
        return predicted_label, scores
    



"""
================================================================================
CLIP ZERO-SHOT: APPROACH 2 (The "Mid-Level" Forward Pass)
================================================================================
WHEN TO USE THIS:
Shift to this approach when you need to show the client/user clean "Confidence 
Percentages" (e.g., 95%) rather than raw cosine similarity scores (e.g., 0.31).
Perfect for triggering business logic (e.g., `if confidence > 0.90: send_alert()`).

THE MECHANICS:
Instead of encoding images and text separately and manually calculating the dot 
product (Approach 1), we pass both directly into the model's forward function. 
CLIP automatically scales the output into logits (using its internal temperature), 
which allows us to apply a standard Softmax to get a 0.0 to 1.0 probability.

QUICK REFERENCE IMPLEMENTATION:
--------------------------------------------------------------------------------
import torch
import clip
from PIL import Image

# 1. Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. Prepare Inputs
image = preprocess(Image.open("target.png")).unsqueeze(0).to(device)
labels = ["a modern kitchen", "a messy bedroom", "an office"]
text = clip.tokenize(labels).to(device)

# 3. Direct Forward Pass
with torch.no_grad():
    # Returns scaled logits
    logits_per_image, _ = model(image, text) 
    
    # Apply Softmax to get percentages (e.g., [0.98, 0.01, 0.01])
    probs = logits_per_image.softmax(dim=-1)[0].tolist() 

# 4. Map back to labels
results = dict(zip(labels, probs))
best_match = max(results, key=results.get)
================================================================================
"""